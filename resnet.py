from keras.datasets import cifar100,cifar10
from keras.preprocessing.image import ImageDataGenerator
import keras
from  datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import random


class ResNet:
    def __init__(self):
        self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_DEPTH = 32, 32, 3
        self.NUM_CLASS = 10
        self.BN_EPSILON = 0.001
        self.WEIGHT_DECAY = 0.0002
        self.batch = 50000
        self.n_epoch = 100  
        self.batch_size = 200
        self.NL = 6
        self.lr_decay = 0.95

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8




    def create_variables(self, name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.WEIGHT_DECAY)

        new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                              regularizer=regularizer)
        return new_variables


    def output_layer(self, input_layer, num_labels):
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = self.create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = self.create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

        fc_h = tf.matmul(input_layer, fc_w) + fc_b
        return fc_h


    def layer_conv(self, input_layer, filters, strides, kernel_size, padding="SAME"):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.WEIGHT_DECAY)
        conv = tf.layers.conv2d(input_layer, 
                                  filters=filters, 
                                  kernel_size=kernel_size, 
                                  strides=strides, 
                                  padding=padding,
                                  kernel_regularizer=regularizer)
        layer = tf.layers.batch_normalization(conv, momentum=0.9)
        re = tf.nn.relu(layer)
        return re


    def residual_basicBlock(self, input_layer, output_channel, _strides=(1, 1)):

        input_channel = input_layer.get_shape().as_list()[-1]

        if input_channel == output_channel:
            shortcut = input_layer
        else:
            shortcut = self.layer_conv(input_layer, filters=output_channel, kernel_size=1 ,strides=(2,2), padding="SAME")
            _strides = (2, 2)

        #layer 1
        x = self.layer_conv(input_layer, filters=output_channel, kernel_size=3 ,strides=_strides, padding="SAME")
        

        #layer 2
        x = self.layer_conv(x, filters=output_channel, kernel_size=3 ,strides=(1, 1), padding="SAME")

        output = x + shortcut
        #output = tf.nn.relu(x)
        return output


    def inference(self, X):
        #first conv
        conv = self.layer_conv(X, filters=16, kernel_size=3 ,strides=[1,1], padding="SAME")
        #conv = tf.nn.relu(conv)

        layers = []
        layers.append(conv)

        for x in range(self.NL):
            layers.append(self.residual_basicBlock(layers[-1], 16))

        for x in range(self.NL):
            layers.append(self.residual_basicBlock(layers[-1], 32))

        for x in range(self.NL):
            layers.append(self.residual_basicBlock(layers[-1], 64))

        conv = layers[-1]

        #global reduce mean
        net = tf.reduce_mean(conv, [1,2])

        with tf.name_scope('fc'):
            logits = self.output_layer(net, self.NUM_CLASS)

        return logits


    def log_dir(self, prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)

        
    def summary(self):
        logdir = self.log_dir("resnet")
        self.file_writer_1 = tf.summary.FileWriter(logdir+"/train", tf.get_default_graph())
        self.file_writer_2 = tf.summary.FileWriter(logdir+"/test", tf.get_default_graph())
        self.write_op = tf.summary.merge_all()


    def prepare_data(self, x_train, y_train, batch_size, ratio=1):
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=.15,
            height_shift_range=.15,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        i=0
        images = []
        labels = []
        stop = int(len(x_train)/batch_size)
        
        x_train_batch = np.array_split(x_train, int(len(x_train)/batch_size))
        y_train_batch = np.array_split(y_train, int(len(x_train)/batch_size))
        
        for datas in datagen.flow(x_train, y_train, batch_size=batch_size):
            images.append(datas[0])
            labels.append(datas[1].reshape(-1))
            
            if i < stop:
              images.append(x_train_batch[i])
              labels.append(y_train_batch[i].reshape(-1))
            
            i += 1
            if i == stop*ratio:
                break
        return zip(images, labels)

    
    def train_loss(self, y, logits):
      # ------- losss -------
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  # entropie croisée
      loss = tf.reduce_mean(xentropy)
      loss_summary = tf.summary.scalar('log_loss', loss)
      # ------- train -------
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(0.001, global_step,
                                         10000, self.lr_decay, staircase=True)

      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      training_op = optimizer.minimize(loss, global_step=global_step)
      # ------- EVAL -------
      correct = tf.nn.in_top_k(logits, y, 1)
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
      accuracy_summary = tf.summary.scalar('accuracy', accuracy)
      
      return training_op, accuracy, loss, learning_rate
      
    
    def train(self, restore=None):
      
        X = tf.placeholder(tf.float32, shape=(None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_DEPTH), name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")
        
        logits = self.inference(X)
        
        training_op, accuracy, loss, learning_rate = self.train_loss(y, logits)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        print ('Start training...')
        print ('----------------------------')

        #self.summary()
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        sess = tf.Session()
        sess.run(init)
        
        for n in range(self.n_epoch):         
            for x_batch, y_batch in self.prepare_data(x_train, y_train, self.batch_size):
               sess.run(training_op, feed_dict={X:x_batch, y:y_batch})


            accuracy_train  = sess.run([accuracy], 
                                             feed_dict={
                                                X: x_train[0:10000], 
                                                y: y_train[0:10000].reshape(-1)
                                            })

            #self.file_writer_1.add_summary(summary, n)
            #self.file_writer_1.flush()

            accuracy_val, loss_val = sess.run([accuracy, loss], 
                feed_dict={X: x_test, y: y_test.reshape(-1)})
            #self.file_writer_2.add_summary(summary, n)   
            #self.file_writer_2.flush()
            print(n, "Training accuracy:", accuracy_train, "Validation accuracy:", accuracy_val, " loss : ", loss_val, " lr : ", learning_rate.eval(session=sess))

            #saver.save(sess, checkpoint_path)

            #if loss_val < best_loss:
                   #saver.save(sess, final_model_path)
                   #best_loss = loss_val


with tf.device("/device:GPU:0"):
  tf.reset_default_graph()
  res = ResNet()
  # Start the training session
  res.train()
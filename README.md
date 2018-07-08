# tensorflow on GPU


### install nvidia

wget -O - -q 'https://gist.githubusercontent.com/allenday/f426e0f146d86bfc3dada06eda55e123/raw/41b6d3bc8ab2dfe1e1d09135851c8f11b8dc8db3/install-cuda.sh' | sudo bash

test :

nvidia-smi



run :

jupyter lab --no-browser --ip=0.0.0.0


tensorboard :

tensorboard --logdir=/home/yannis/project/tf_logs

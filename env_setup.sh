#!/bin/bash

if [[ `uname` == "Linux" ]]; then
    echo "Installing dependencies for Linux environment: anaconda, tensorflow, python and related libs."

	# # install cuda
	# wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
	# sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
	# rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
	# sudo apt-get update
	# sudo apt-get install -y cuda

	# # get cudnn
	# CUDNN_FILE=cudnn-7.0-linux-x64-v4.0-prod.tgz
	# wget http://developer.download.nvidia.com/compute/redist/cudnn/v4/${CUDNN_FILE}
	# tar xvzf ${CUDNN_FILE}
	# rm ${CUDNN_FILE}
	# sudo cp cuda/include/cudnn.h /usr/local/cuda/include # move library files to /usr/local/cuda
	# sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	# sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
	# rm -rf cuda

	# # install anaconda
	# wget http://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh
	# bash Anaconda2-4.0.0-Linux-x86_64.sh
	# rm Anaconda2-4.0.0-Linux-x86_64.sh
	# echo 'export PATH="/home/$USER/anaconda2/bin:$PATH"' >> ~/.bashrc

	# # set the appropriate library path
	# echo 'export CUDA_HOME=/usr/local/cuda
	# export CUDA_ROOT=/usr/local/cuda
	# export PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
	# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
	# ' >> ~/.bashrc

	# install tensorflow under anaconda virtual environment
	# source ~/.bashrc
	# if [[ ! -d /home/${USER}/anaconda2/envs/tensorflow ]]; then
	# 	conda create -p  /home/${USER}/anaconda2/envs/tensorflow python=2.7
	# fi
	# ls /home/${USER}/anaconda2/envs/tensorflow
	# source activate tensorflow

	pip install --ignore-installed --upgrade --requirement config/requirements-linux-gpu.txt


elif [[ `uname` == "Darwin" ]]; then
    echo "Installing dependencies for Mac environment: anaconda, tensorflow, python and related libs."
    pip install --ignore-installed --upgrade --requirement config/requirements-mac-cpu.txt

fi



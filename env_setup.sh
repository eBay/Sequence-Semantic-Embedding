#!/bin/bash

# install python

sudo apt-get update
sudo apt-get install python-pip python-dev build-essential python-tk
sudo easy_install pip

# install dependent python packages

pip install -r requirements.txt


#!/bin/bash

# Install SentencePiece (https://github.com/google/sentencepiece/blob/master/README.md)
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v

# Create a virtual environment and install all the necessary dependencies
mkvirtualenv interl
workon interl
pip install -r requirements.txt
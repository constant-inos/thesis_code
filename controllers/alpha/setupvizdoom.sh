apt-get update

apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip
it
sudo apt-get install libboost-all-dev
sudo apt-get install python3-dev python3-pip
sudo apt-get install liblua5.1-dev

# go to your user folder
cd ~
# get julia
wget https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.0-linux-x86_64.tar.gz
# extract the file (eXtract File as options)
tar xf julia-1.3.0-linux-x86_64.tar.gz
# Create a shortcut (a soft link) that's places in a globally accessible folder
sudo ln -s ~/julia-1.3.0/bin/julia /usr/local/bin/julia

pip install vizdoom
pip install varname


julia

using Pkg
Pkg.add("CxxWrap")
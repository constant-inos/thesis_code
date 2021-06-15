#!/bin/bash

FILE=webots_2021a_amd64.deb

if test -f "$FILE"; then
	echo "deb exists"
else
	curl -L -O https://github.com/cyberbotics/webots/releases/download/R2021a/webots_2021a_amd64.deb
fi


#curl -L -O https://github.com/cyberbotics/webots/releases/download/R2021a/webots_2021a_amd64.deb

#####################################3

if [[ $EUID -ne 0 ]]; then
       echo "This script must be run as root"
       exit 1
fi

apt update
apt install --yes lsb-release g++ make libavcodec-extra libglu1-mesa libxkbcommon-x11-dev execstack libusb-dev libxcb-keysyms1 libxcb-image0 libxcb-icccm4 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcomposite-dev libxtst6 libnss3
if [[ -z "$DISPLAY" ]]; then
       apt install --yes xvfb
fi

UBUNTU_VERSION=$(lsb_release -rs)
if [[ $UBUNTU_VERSION == "16.04" ]]; then
       apt install --yes libav-tools
elif [[ $UBUNTU_VERSION == "18.04" ]]; then
       apt install --yes ffmpeg
elif [[ $UBUNTU_VERSION == "20.04" ]]; then
       apt install --yes ffmpeg
else
       echo "Unsupported Linux version."
fi

######################################

sudo apt-get install libjxr0
sudo apt-get install libraw16
sudo apt-get install libfreeimage3
sudo apt-get install libzzip-0-13
sudo apt-get install libssh-4
sudo apt-get install libssh-dev
sudo apt-get install libzip4
sudo apt-get install libzip-dev
apt install python3.6-gdbm

######################################

sudo dpkg -i webots_2021a_amd64.deb

######################################

export WEBOTS_HOME=/snap/webots/current/usr/share/webots
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller

#########################################

# cd '/content/gdrive/MyDrive/Colab Notebooks/thesis_code'
# sudo xvfb-run --auto-servernum webots --mode=fast --stdout --stderr --minimize --batch --no-sandbox ./worlds/Eworld.wbt

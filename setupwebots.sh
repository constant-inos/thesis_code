wget -qO- https://cyberbotics.com/Cyberbotics.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://cyberbotics.com/debian/ binary-amd64/'
sudo apt-get update
sudo bash /content/gdrive/MyDrive/thesis_code/webots/depend.sh
sudo apt-get install webots
export WEBOTS_HOME=/snap/webots/current/usr/share/webots
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller
pip install varname
cd ./webots
sudo xvfb-run --auto-servernum webots --mode=fast --stdout --stderr --minimize --batch --no-sandbox ./worlds/Cworld.wbt
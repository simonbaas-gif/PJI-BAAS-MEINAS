## Python

### NumPy

sudo apt-get install python-numpy
OR
pip install numpy
pip3 install numpy

### Scipy

sudo apt-get install python-scipy

### Matplotlib

sudo apt-get install python-matplotlib

OR

python2
pip install matplotlib

python3
pip3 install matplotlib

### PIL for Python

python2
sudo apt-get install python-pip python-dev
sudo pip install pillow

python3
sudo apt-get install python3-pip python3-dev
sudo pip3 install pillow

OR

sudo apt-get install python3-pil

### Intertools

python2
pip install more-itertools

python3
pip3 install more-itertools

### Mahotas

python2
pip install mahotas

python3
pip3 install mahotas

### PIP

sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtwebkit4 libqt4-test
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

### Tensorflow

For CPU
pip install tensorflow

For GPU
pip install tensorflow-gpu

### OpenCV

python2
sudo apt-get install python-dev python-numpy

python3
sudo apt-get install python3-dev python3-numpy

sudo apt-get install python-opencv

sudo pip install opencv-contrib-python
sudo pip install "picamera[array]"
sudo pip install imutils
sudo pip install pyautogui

sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4

Install OpenCV : https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/

### PyQt5

python2
sudo apt-get install python-pyqt5

python3
sudo apt-get install python3-pyqt5

## COCO API installation

To install:
-For Matlab, add coco/MatlabApi to the Matlab path (OSX/Linux binaries provided)
-For Python, run "make" under coco/PythonAPI
-For Lua, run “luarocks make LuaAPI/rocks/coco-scm-1.rockspec” under coco/

### Pycocotools

pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

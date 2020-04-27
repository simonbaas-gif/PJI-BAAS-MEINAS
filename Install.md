# Installation du projet

Pour le projet nous allons nous servir de l'API fournie par Google (TensorFlow Object Detection API)
On a besoin des models prédéfinis par tensorflow :
```
git clone https://github.com/tensorflow/models
```

Nous avons besoin de pip :
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```

Ensuite on se sert de pip afin d'installer les librairies suivantes :
```
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

Puis nous avons besoin de l'API de COCO car celle ci fournie une large base de données d'images :
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

L'API utilise des fichiers .proto. Ces fichiers sont compilés en fichiers python.
Google fournit un logiciel afin de faire cela, Protobuf.

Il faut donc le télécharger et l'extraire dans tensorflow/models/research/
```
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```

Puis on peut lancer la commande afin de compiler (depuis tensorflow/models/research) :
```
./bin/protoc object_detection/protos/*.proto --python_out=.
```

Ensuite il faut ajouter les librairies à PYTHONPATH (depuis tensorflow/models/research) :
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Il faut ensuite lancer le setup.py (depuis tensorflow/models/reasearch) :
```
python setup.py build
python setup.py install
```

Il est possible d'exécuter cette commande afin de voir si l'installation a bien était effectuée :
```
python object_detection/builders/model_builder_test.py
```

Ou bien d'exécuter le tutoriel sous jupyter :
```
jupyter notebook object_detection_tutorial.ipynb
```

## Quelques autres librairies python

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

python
pip install matplotlib

python3
pip3 install matplotlib

### PIL for Python

python
sudo apt-get install python-pip python-dev
sudo pip install pillow

python3
sudo apt-get install python3-pip python3-dev
sudo pip3 install pillow

OR

sudo apt-get install python3-pil

### Intertools

python
pip install more-itertools

python3
pip3 install more-itertools

### Mahotas

python
pip install mahotas

python3
pip3 install mahotas

### PIP

sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtwebkit4 libqt4-test
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

### TensorFlow

For CPU
pip install tensorflow

For GPU
pip install tensorflow-gpu

### NumPy

python
sudo apt-get install python-dev python-numpy

python3
sudo apt-get install python3-dev python3-numpy

### OpenCV

python
sudo apt-get install python-opencv

python3
sudo apt-get install python-opencv

Install OpenCV : https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/

### PyQt5

python
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

# Introduction
This project partially implement SfSNet project. 
I will implement all codes in SfSNet project in the future,
if I have enough time.

Currently implemented:
* Implement test_SfSNet.m as SfSNet_test.py
* Implement functions/*.m in src/functions.py
* move some constant variables to config.py 

If there are bugs in SfSNet_test.py, please open a issue.

# Dependencies
* Python libs in requirements.txt

# Run SfSNet_test.py
* Download shape_predictor_68_face_landmarks.dat from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 , 
and uncompress it to directory `data`.

* Create and activate a virtual environment
        * for python 2.*
    ```bash
    pip install virtualenv
    virtualenv -p python2 venv2
    source venv2/bin/activate
    ```
    * for python 3.*
    ```bash
    pip install virtualenv
    virtualenv -p python3 venv3
    source venv3/bin/activate
    ```
* Install python dependencies using command:
    ```bash
    pip install -r requirements.txt
    ```
* Put your test images in 'Images', and 
    ```bash
    python SfSNet_test.py
    ```

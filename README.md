# Introduction
This project partially implement SfSNet project. 

Currently implemented:
* Implement test_SfSNet.m as SfSNet_test.py
* Implement functions/*.m in src/functions.py
* move some constant variables to config.py 
* SfSNet_train in train.py

If there are any bugs, please open a issue.

# Dependencies
* Python libs in requirements.txt

# SfSNet_test.py
* Download shape_predictor_68_face_landmarks.dat from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 , 
and uncompress it to project_dir/data.

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
* Deactivate virtual environment
    ```bash
    deactivate
    ```
# Train
* Activate the same virtual environment created 
before if it does not been activated.
* Download the SfSNet(synthetic) and CELABA(real) dataset from 
the original SfSNet project and uncompress them.

* Modify `SFSNET_DATASET_DIR` and `SFSNET_DATASET_DIR_NPY` in 
config.py to your directory.

* Modify `CELABA_DATASET_DIR` and `CELABA_DATASET_DIR_NPY` in 
config.py to your directory.

* Train SfSNet is split into two stages(please read the section
3.1 of SfSNet paper). Stage 0 is train SfSNet with synthetic 
dataset, then using the weights trained in stage to preprocess
a real dataset CELABA. Stage 1 is train SfSNet with real and 
synthetic dataset.
    * Stage 0  
        1. Run preprocess_dataset.py to preprocess synthetic dataset
            ```bash
            python preprocess_dataset.py --stage 0
            ```
            the size of processed data is about 151.3GB.

        2. Run train.py
            ```bash
            python train.py --stage 0
            ```
            if you press CTRL+C, the weights of current model will
             be saved to ./data.
    * Stage 1  
        1. Run preprocess_dataset.py to preprocess real(CELABA) dataset
            ```bash
            python preprocess_dataset.py --stage 1 --weights data/weights_2019.04.19_19.00.10.pth
            ```
            the size of processed data is about 141.7GB.

        2. Run train.py
            ```bash
            python train.py --stage 1
            ```
            if you press CTRL+C, the weights of current model will be 
            saved to ./data.
    
# Predict

if you train model with train.py, you should eval 
your model with predict.py.

* Activate the same virtual environment created before if it does not
been activated.
* Put your image in directory `Images`, then
    ```bash
    python predict.py --weights data/weights_2019.04.20_08.47.51.pth
    ```
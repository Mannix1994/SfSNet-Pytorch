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
    ```bash
    pip install virtualenv
    virtualenv venv
    source venv/bin/activate
    ```
* Install python dependencies using command:
    ```bash
    pip install -r requirements.txt
    ```
* Put your test images in 'Images', and 
    ```bash
    python SfSNet_test.py
    ```

# Train
* Download the SfSNet from the original SfSNet project and
uncompress it to a directory

* Modify `SFSNET_DATASET_DIR` and `SFSNET_DATASET_DIR_NPY` in 
config.py to your directory.

* Run preprocess_dataset.py
    ```bash
    python preprocess_dataset.py
    ```
    the size of processed data is about 147.6GB.

* Run train.py
    ```bash
    python train.py
    ```
    if you press CTRL+C, the weights of current model will be 
    saved to ./data.
    
# Predict

* Put your image in directory `Images`, then
    ```bash
    python predict.py
    ```
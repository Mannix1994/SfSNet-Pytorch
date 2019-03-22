# Introduction
This project partially implement SfSNet project. 
I will implement all codes in SfSNet project in the future,
if I have enough time.

Currently implemented:
* 
* Implement test_SfSNet.m as _test_SfSNet.py
* Implement functions/*.m in functions.py
* move some constant variables to config.py 

If there are bugs in _test_SfSNet.py, please open a issue.

# Dependencies
* Python 2.7
* Caffe with pycaffe module(read build-caffe.md)
* Python libs in requirements.txt

# Run _test_SfSNet.py

You should modify the variable CAFFE_ROOT in config.py 
to your own caffe install directory. Then, install python
dependencies using command:
```bash
pip install -r requirements.txt
```
then put your test images in 'Images', and 
```bash
python SfSNet_test.py
```
and input 0 when the program tell you to input. the results
will be saved in directory 'result' 

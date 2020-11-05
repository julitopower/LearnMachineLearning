# ML Refresher example notebooks

These notebooks are inspired by the content of the fantastic book "[Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 2nd Edition ](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646/ref=sr_1_3?crid=3SRL70QAD46FM)" by [Aurélien Geron](https://twitter.com/aureliengeron). This book is a goldmine that you should buy.

# Environment setup

We provide a conda environment configuration that defines all the dependenies needed to run these example notebooks. To create a new conda environment simply type:

```
conda env create -f=environment.yml 
```

## Installing coda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
```

Verify the installation by executing

```
conda env list
```

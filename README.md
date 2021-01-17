# Pyramid_Scale_Network
This is the PyTorch version repo for "Exploit the potential of Multi-column architecture for Crowd Counting", which delivered a state-of-the-art, straightforward and end-to-end architecture for crowd counting tasks. We also recommend another work on crowd counting([Deep Density-aware Count Regressor](https://github.com/GeorgeChenZJ/deepcount)), which is accepted by ECAI2020.

# Datasets
ShanghaiTech Dataset

# Prerequisites
We strongly recommend Anaconda as the environment.  
  
Python: 3.6  
  
PyTorch: 1.5.0

# Train & Test
1、python make_dataset.py # generate the ground truth. the ShanghaiTech dataset should be placed in the "datasets" directory.  
2、python train.py # train model  
3、python val.py # test model

# Results
partA: MAE 55.5 MSE 90.1  
  
partB: MAE 6.8 MSE 10.7

# MIRAGE
MIRAGE is a human trajectory generative model designed as an intensity-free neural Temporal Point Process integrating a neural Exploration and Preferential Return model to imitate the human decision-making process in trajectory generation.

## How to run the code
```
cd code
sh run.py
```
The default dataset is NewYork, you can change the dataset in run.sh to train and evaluate on Istanbul and Tokyo datasets.  
Our detailed parameters for all datasets are in /code/parameters.py

## Requirements
Python: 3.8.18  
torch: 1.12.1  
recbole==1.2.0  
numpy  
matplotlib  
scikit-learn  
setproctitle  
ray  
kmeans-pytorch  
pyarrow  
ray[tune]  
GPU with CUDA 11.3  

## Reference
If you use our code or datasets, please cite:

## Note
The implementation is based on [LogNormMix](https://github.com/shchur/ifl-tpp).

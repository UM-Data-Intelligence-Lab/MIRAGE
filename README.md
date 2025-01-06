# MIRAGE
MIRAGE is a human trajectory generative model designed as an intensity-free neural Temporal Point Process integrating a neural Exploration and Preferential Return model to imitate the human decision-making process in trajectory generation. Please see the details in our paper below:  
- Bangchao Deng, Xin Jing, Tianyue Yang, Bingqing Qu, Dingqi Yang*, Philippe Cudre-Mauroux, Revisiting Synthetic Human Trajectories: Imitative Generation and Benchmarks Beyond Datasaurus, In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'25), Aug. 2025, Toronto.
  
## How to run the code
```
cd code
sh run.py
```
The default dataset is NewYork, you can change the dataset in run.sh to train and evaluate on Istanbul and Tokyo datasets.  
Our detailed parameters for all datasets are in /code/parameters.py

## Requirements
```
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
```

## Reference
If you use our code or datasets, please cite:
```
@article{deng2024revisiting,
  title={Revisiting Synthetic Human Trajectories: Imitative Generation and Benchmarks Beyond Datasaurus},
  author={Deng, Bangchao and Jing, Xin and Yang, Tianyue and Qu, Bingqing and Cudre-Mauroux, Philippe and Yang, Dingqi},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={},
  year={2025}
}
```
## Note
The implementation is based on [LogNormMix](https://github.com/shchur/ifl-tpp) and our Task-based Evaluation Protocol LocRec and NexLoc are based on [RecBole](https://github.com/RUCAIBox/RecBole).

#!/bin/bash

python -u train.py --dataset NewYork 
python evaluation.py --datasets NewYork --task SemLoc 
python evaluation.py --datasets NewYork --task EpiSim 
python evaluation.py --datasets NewYork --task Stat 
python evaluation.py --datasets NewYork --task LocRec 
python evaluation.py --datasets NewYork --task NexLoc 

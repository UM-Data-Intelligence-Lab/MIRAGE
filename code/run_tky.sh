#!/bin/bash

python -u train.py --dataset Tokyo 
python evaluation.py --datasets Tokyo --task SemLoc &
python evaluation.py --datasets Tokyo --task EpiSim &
python evaluation.py --datasets Tokyo --task Stat &
python evaluation.py --datasets Tokyo --task LocRec --cuda 2 &
python evaluation.py --datasets Tokyo --task NexLoc --cuda 2 &
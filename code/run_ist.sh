#!/bin/bash

python -u train.py --dataset Istanbul 
python evaluation.py --datasets Istanbul --task SemLoc &
python evaluation.py --datasets Istanbul --task EpiSim &
python evaluation.py --datasets Istanbul --task Stat &
python evaluation.py --datasets Istanbul --task LocRec --cuda 1 &
python evaluation.py --datasets Istanbul --task NexLoc --cuda 1 &
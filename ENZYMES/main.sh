#!/bin/bash
job_cmd1='python -m node2vec.src.main --input ../ENZYMES295.edgelist --output ../../ENZYMES/datasets/node2vec_emb/ENZYMES295.emb'
job_cmd2='python -m graphwave.src.main --input ../data/ENZYMES295.csv --output ../../ENZYMES/datasets/graphwave/ENZYMES295.csv'
job_cmd3='python -m main --dataset E295'

eval $job_cmd1
eval $job_cmd2
eval $job_cmd3

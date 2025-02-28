#!/bin/bash
python train.py --folder "data" --out "logs" --num-steps 5 --num-shots 2  --num-ways 5 --num-batches 1000 --batch-size 16 --step-size 0.005 --use-cuda --num-workers 8

#python train_orig.py --folder "data" --out "logs" --num-shots 5 --num-ways 5 --num-batches 100 --batch-size 16 --use-cuda --num-workers 8


python test.py --folder "data" --out "logs" --num-steps 20 --num-shots 2 --num-ways 5 --num-batches 100 --batch-size 16 --step-size 0.001 --use-cuda --num-workers 8

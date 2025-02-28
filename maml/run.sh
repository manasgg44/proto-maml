#!/bin/bash
python train.py --folder "data" --out "logs" --download --num-steps 1 --num-shots 5 --num-ways 5 --num-batches 100000 --batch-size 16 --step-size 0.0005 --use-cuda --num-workers 8
#python test.py --folder "data" --out "logs" --download --num-steps 20 --num-shots 5 --num-ways 5 --num-batches 5 --batch-size 16 --step-size 0.001 --use-cuda --num-workers 8

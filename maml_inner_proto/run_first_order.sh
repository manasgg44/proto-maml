#!/bin/bash
#python train.py --folder "data" --out "logs_first_order" --download --num-steps 1 --num-shots 5 --num-ways 5 --num-batches 100 --batch-size 16 --step-size 0.0005 --use-cuda --num-workers 8 --first-order
python test.py --folder "data" --out "logs_first_order" --download --num-steps 15 --num-shots 5 --num-ways 5 --num-batches 10 --batch-size 16 --step-size 0.0001 --use-cuda --num-workers 8 --first-order

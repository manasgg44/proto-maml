#!/bin/bash
#python trainn.py --folder "data" --out "logs" --num-shots 5 --num-ways 5 --num-batches 1000 --batch-size 16 --use-cuda --num-workers 8
python test.py --folder "data" --out "logs" --download --num-shots 5 --num-ways 5 --num-batches 5 --batch-size 16 --use-cuda --num-workers 8

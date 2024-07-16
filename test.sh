#!/bin/bash

start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python test.py \
--model_path ./ckpt/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/20240716_084420/checkpoint-50

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Elapsed time: ${elapsed_time} seconds"

#!/bin/bash

start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python ./test.py \
--model_path /data2/hyunwook/dacon/ckpt/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/20240727_120436/checkpoint-100

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

echo "Elapsed time: ${elapsed_time} seconds"

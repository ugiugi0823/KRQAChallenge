CUDA_VISIBLE_DEVICES=0 torchrun  --nproc_per_node=1 \
./peft/train_test1.py \
--train_path ./data/train.csv \
--model_id Qwen/Qwen2-72B-Instruct
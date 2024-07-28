import torch, os, pytz, re
import argparse
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from datetime import datetime
from transformers.integrations import WandbCallback
from sklearn.metrics import f1_score
import numpy as np
import bitsandbytes as bnb
import torch.distributed as dist


import string
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning, message=".*TokenIndexer.*")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank

def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Original padding side for {model_id}: {tokenizer.padding_side}")
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_id, local_rank):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True
    )
    return model

def main(args):
    local_rank = setup_distributed()
    
    # model_id = "nlpai-lab/KULLM3"
    # model_id ="meta-llama/Meta-Llama-3-70B-Instruct"
    model_id = args.model_id
    # model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

    asia_timezone = pytz.timezone('Asia/Seoul')
    start_time = datetime.now(asia_timezone).strftime("%Y%m%d_%H%M%S")
    output_dir_ = f"./ckpt/{model_id}/{start_time}" 

    if local_rank == 0:
        print("이것은 마스터 프로세스입니다.")
        os.makedirs(output_dir_, exist_ok=True)
        script_name = os.path.basename(__file__)
        script_path = os.path.join(output_dir_, f"{start_time}_{script_name}")
        with open(__file__, 'r') as source_file, open(script_path, 'w') as target_file:
            target_file.write(source_file.read())
        print(f"스크립트가 {script_path}에 저장되었습니다.")
        
       

    dataset = load_dataset('csv', data_files={'train': args.train_path})
    dataset = dataset['train'].train_test_split(test_size=10, shuffle=True)
    
    train_dataset = dataset['train'].shuffle(seed=42).select(range(2700))
    print("🧐"*40)
    print(len(train_dataset))
    
    print("🧐"*40)
    val_dataset = dataset['test']

    if local_rank == 0:
        print("Train Data Columns:", train_dataset.column_names)
        print("Train Data Sample:", train_dataset[0])

    model = load_model(model_id, local_rank)
    tokenizer = setup_tokenizer(model_id)

    def preprocess_function(examples):
        prompts = [f"Context: {context}\nQuestion: {question}\nAnswer: {answer}" 
                   for context, question, answer in zip(examples['context'], examples['question'], examples['answer'])]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=1024)

    columns_to_remove = train_dataset.column_names
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove, num_proc=4)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove, num_proc=4)

    if local_rank == 0:
        print("Processed Train Dataset Sample:", train_dataset[0])

    model = prepare_model_for_kbit_training(model)
    # for name, module in model.named_modules():
    #     print(name)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=output_dir_,
        num_train_epochs=3,
        learning_rate=5e-2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1, 
        warmup_steps=5000,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        gradient_accumulation_steps=1,
        report_to=["wandb"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        local_rank=local_rank,
        ddp_find_unused_parameters=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    def normalize_answer(s):
        def remove_(text):
            ''' 불필요한 기호 제거 '''
            text = re.sub("'", " ", text)
            text = re.sub('"', " ", text)
            text = re.sub('《', " ", text)
            text = re.sub('》', " ", text)
            text = re.sub('<', " ", text)
            text = re.sub('>', " ", text)
            text = re.sub('〈', " ", text)
            text = re.sub('〉', " ", text)
            text = re.sub("\(", " ", text)
            text = re.sub("\)", " ", text)
            text = re.sub("'", " ", text)
            text = re.sub("'", " ", text)
            return text
        def white_space_fix(text):
            '''연속된 공백일 경우 하나의 공백으로 대체'''
            return ' '.join(text.split())
        def remove_punc(text):
            '''구두점 제거'''
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            '''소문자 전환'''
            return text.lower()
        return white_space_fix(remove_punc(lower(remove_(s))))

    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        # 문자 단위로 f1-score를 계산 합니다.
        prediction_Char = []
        for tok in prediction_tokens:
            now = [a for a in tok]
            prediction_Char.extend(now)
        ground_truth_Char = []
        for tok in ground_truth_tokens:
            now = [a for a in tok]
            ground_truth_Char.extend(now)
        common = Counter(prediction_Char) & Counter(ground_truth_Char)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_Char)
        recall = 1.0 * num_same / len(ground_truth_Char)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        vocab_size = tokenizer.vocab_size
        f1_scores = []
        
        for pred, label in zip(predictions, labels):
            valid_pred = pred[label != -100]
            valid_label = label[label != -100]
            
            valid_pred = np.clip(valid_pred, 0, vocab_size - 1)
            valid_label = np.clip(valid_label, 0, vocab_size - 1)
            
            try:
                decoded_pred = tokenizer.decode(valid_pred, skip_special_tokens=True)
                decoded_label = tokenizer.decode(valid_label, skip_special_tokens=True)
                
                f1 = f1_score(decoded_pred, decoded_label)
                f1_scores.append(f1)
            except Exception as e:
                print(f"Error in processing or F1 calculation: {e}")
        
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
        else:
            avg_f1 = 0
        
        return {"eval_f1": avg_f1}



    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[WandbCallback()],
        compute_metrics=compute_metrics,
    )

    if local_rank == 0:
        print("⛴️"*40)
        print(model_id)
        print("⛴️"*40)
    
    trainer.train()

    if local_rank == 0:
        trainer.save_model(f"{output_dir_}/final_model")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default='./data/train.csv', help="Train Path")
    p.add_argument("--model_id", type=str, default='./data/train.csv', help="Train Path")
    args = p.parse_args()
    main(args)
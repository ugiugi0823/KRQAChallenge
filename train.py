import torch, os, pytz
import argparse
import warnings
import random
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
    
    # model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    # model_id = "nlpai-lab/KULLM3"
    model_id = "google/gemma-2-9b-it"

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
    train_dataset = dataset['train'].shuffle(seed=42).select(range(2500))
    val_dataset = dataset['test']

    if local_rank == 0:
        print("Train Data Columns:", train_dataset.column_names)
        print("Train Data Sample:", train_dataset[0])

    model = load_model(model_id, local_rank)
    tokenizer = setup_tokenizer(model_id)

    def preprocess_function(examples):
        instructions = [
            "Please answer briefly in one sentence.",
            "Provide a concise answer.",
            "Answer in no more than two sentences.",
            "Keep your response short and to the point."
        ]
        
        prompts = [f"Context: {context}\nQuestion: {question}\nInstructions: {random.choice(instructions)}\nAnswer: {answer}" 
                   for context, question, answer in zip(examples['context'], examples['question'], examples['answer'])]
        return tokenizer(prompts, truncation=True, padding="max_length", max_length=1024)

    columns_to_remove = train_dataset.column_names
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove, num_proc=4)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove, num_proc=4)

    if local_rank == 0:
        print("Processed Train Dataset Sample:", train_dataset[0])

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = TrainingArguments(
        output_dir=output_dir_,
        num_train_epochs=5,
        learning_rate=5e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1, 
        warmup_steps=500,
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
        gradient_accumulation_steps=8,
        report_to=["wandb"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        local_rank=local_rank,
        ddp_find_unused_parameters=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        valid_preds = predictions[labels != -100]
        valid_labels = labels[labels != -100]
        vocab_size = tokenizer.vocab_size
        valid_preds = np.clip(valid_preds, 0, vocab_size - 1)
        valid_labels = np.clip(valid_labels, 0, vocab_size - 1)
        try:
            decoded_preds = tokenizer.batch_decode(valid_preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
            f1 = f1_score(decoded_labels, decoded_preds, average='macro')
        except Exception as e:
            print(f"Decoding error: {e}")
            f1 = 0
        return {"eval_f1": f1}

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
        max_seq_length=1024,
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
    args = p.parse_args()
    main(args)

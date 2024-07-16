import os, pytz, argparse
import json
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
from datetime import datetime

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

def generate_response(model, tokenizer, question_prompt):
    inputs = tokenizer(question_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 응답에서 "Answer:" 이후의 텍스트만 추출
    if "Answer:" in response:
        response = response.split("Answer:", 1)[1].strip()
        # 첫 문장 또는 최대 30단어만 유지
        response = ' '.join(response.split()[:30])
        # 추가 지시사항이나 다음 질문의 시작점 제거
        for cut_point in ["Question:", "Context:", "Instructions:", "⊙"]:
            if cut_point in response:
                response = response.split(cut_point, 1)[0]
    return response.strip()

def main(args):
    peft_model_path = args.model_path
    path_parts = peft_model_path.split('/')
    
    print("🌊"*40)
    model_id = f"{path_parts[5]}/{path_parts[6]}"
    print(model_id)
    print("🌊"*40)
    
    # 기본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA 가중치 로드
    model = PeftModel.from_pretrained(model, peft_model_path)
    
    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    # 테스트 데이터 로드
    test_data = pd.read_csv('./data/test.csv')
    
    # 모델 추론
    submission_dict = {}
    
    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            context = row['context']
            question = row['question']
            id = row['id']
            if context is not None and question is not None:
                # 지시사항을 포함한 프롬프트 생성
                instruction = "Please answer briefly in one or two sentences."
                question_prompt = f"Context: {context}\nQuestion: {question}\nInstructions: {instruction}\nAnswer:"
                answer = generate_response(model, tokenizer, question_prompt)
                
                # NaN 또는 null 값 체크
                if pd.isna(answer) or answer == '' or answer is None:
                    print("🧊🧊🧊🧊 Not good!🧊🧊🧊🧊")
                    # 한 번 더 시도
                    answer = generate_response(model, tokenizer, question_prompt)
                
                submission_dict[id] = answer if not pd.isna(answer) and answer != '' and answer is not None else 'No valid answer'
            else:
                submission_dict[id] = 'Invalid question or context'
        except Exception as e:
            print(f"Error processing question {id}: {e}")
            submission_dict[id] = 'Error occurred'
    
    # 제출 파일 생성
    df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
    
    asia_timezone = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(asia_timezone).strftime("%Y%m%d_%H%M%S")
    
    submission_folder = f'./data/sub/{model_id}'
    os.makedirs(submission_folder, exist_ok=True)
    peft_model_dir_name = os.path.basename(peft_model_path)
    submission_filename = os.path.join(submission_folder, f'sub_{current_time}_{path_parts[5]}_{path_parts[6]}_{path_parts[7]}_{peft_model_dir_name}.csv')
    
    df.to_csv(submission_filename, index=False)
    print(f"Submission file created: {submission_filename}")
    
    # NaN 또는 null 값 체크
    nan_count = df['answer'].isna().sum()
    null_count = (df['answer'] == '').sum() + (df['answer'] == 'No valid answer').sum() + (df['answer'] == 'Error occurred').sum()
    
    print(f"NaN 값의 개수: {nan_count}")
    print(f"Null 또는 무효한 값의 개수: {null_count}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default='str', help="Train Path")
    args = p.parse_args()
    main(args)

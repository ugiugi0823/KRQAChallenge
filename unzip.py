import zipfile
import os

# 압축 해제할 ZIP 파일 경로
zip_file_path = './data/open.zip'

# 압축 해제할 디렉터리 경로
extract_to_path = './data/'

# 디렉터리가 존재하지 않으면 생성
os.makedirs(extract_to_path, exist_ok=True)

# ZIP 파일 열기
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 모든 파일 압축 해제
    zip_ref.extractall(extract_to_path)

print(f'Files have been extracted to {extract_to_path}')

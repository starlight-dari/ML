# 1. 사용할 기본 이미지 선택 (Python 3.9)
FROM python:3.8.20

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 패키지 설치 및 시스템 패키지 업데이트
RUN apt update && apt install -y libgl1 python3-pip && \
    pip install --no-cache-dir gdown

# 4. 로컬 파일들을 컨테이너로 복사
COPY . .

# 5. 필요한 Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 실행 스크립트 복사 및 실행 권한 부여
COPY download_and_run.sh /app/download_and_run.sh
RUN chmod +x /app/train_dreambooth.py /app/train_dreambooth_lora.py /app/download_and_run.sh

# 6. 컨테이너 실행 시 수행할 명령
CMD ["/bin/bash", "-c", "/app/download_and_run.sh && pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git && python api_ml.py"]




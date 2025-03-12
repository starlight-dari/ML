# 1. 사용할 기본 이미지 선택 (Python 3.9)
FROM python:3.8.20

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# Google Drive 파일 다운로드를 위한 gdown 설치
RUN pip install gdown

# 3. 로컬 파일들을 컨테이너로 복사
COPY . .

# 4. 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update && apt install -y libgl1

# 실행 스크립트 복사 및 실행 권한 부여
COPY download_and_run.sh /app/download_and_run.sh
RUN chmod +x /app/download_and_run.sh

# 컨테이너 실행 시 스크립트 실행
CMD ["/bin/bash", "-c", "
    chmod +x /app/download_and_run.sh && \
    chmod +x /app/train_dreambooth.py && \
    /app/download_and_run.sh && \
    python api_ml.py
"]




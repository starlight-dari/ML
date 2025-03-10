# 1. 사용할 기본 이미지 선택 (Python 3.9)
FROM python:3.8.20

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 로컬 파일들을 컨테이너로 복사
COPY . .

# 4. 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update && apt install -y libgl1

# 5. 실행할 기본 명령어 설정
CMD ["python", "api_ml.py"]


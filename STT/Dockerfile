# 1. 파이썬 3.12 슬림 버전 사용
FROM python:3.12-slim-bookworm

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 필수 패키지 설치 (오디오 처리를 위한 ffmpeg 등)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 결과 저장용 폴더 미리 생성
RUN mkdir -p /app/input /app/Audio_result /app/suspicious_segments

# 7. 포트 개방
EXPOSE 8000

# 8. 서버 실행
CMD ["python", "api.py"]
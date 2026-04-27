## STT 과정
- 음성 입력데이터 전처리 (FRCRN, Denoisere, DCCRN)
- 음성 STT작업 (Faster-Whisper) + NVIDIA Canary
- AI 판별 모델 (AASIST, RawNet2 - 성능 저조할 시 사용(복잡))
  - Hugging Face를 통해 사전 학습된 모델 가져다 사용 (테스트)
- 피싱 판별
  - Hugging Face에 명확한 모델 X
  - 따라서 LLM에 위임 vs 직접 데이터를 구해 파인튜닝
    - 데이터 출처: 금융감독원 '그놈 목소리' 사이트에 공개된 보이스피싱 스크립트(텍스트)나 공공데이터포털의 스미싱 데이터


## 사용 방법

- 도커 실행
```
  docker-compose up
```

- 실행 후 출력값
1. Audio_result 폴더 내 
  - analysis_result.json : JSON 형식 결과
  - analysis_summary.txt : 인간이 읽기 좋은 결과
2. suspicious_segmenets: AI물어보기용 분기별 AI 판별 결과 (사용 할지말지는 마음대로)


- JSON 형식
  
| 필드명 | 타입 | 설명 |
|--------|------|------|
| input_file | string | 입력된 음성 파일 이름 |
| stt_text | string | 음성 → 텍스트 변환 결과 |
| ai_probability | float | 전체 AI 음성 의심 확률 (%) |
| final_label | string | 최종 판정 결과 |
| suspicious_segments | array | 의심 구간 리스트 |

- txt 형식
  
| 필드명 | 타입 | 설명 |
|--------|------|------|
| start_sec | float | 구간 시작 시간 (초) |
| end_sec | float | 구간 종료 시간 (초) |
| fake_score | float | 해당 구간의 AI 의심 확률 (%) |



## 전체 로직: FFmepg -> Whisper -> Wav2Vec2 -> DSP 기반 Feature Engineering

- 입력 음성에 대한 확장자 처리 (.m4a, etc... -> .wav)
  - FFmpeg 변환 처리 (데이터 정규화)

- 음성 데이터 STT 작업
  - Whisper (OpenAI) 사용
  - faster-whisper로 구현 (다국어 지원 및 노이즈에 강력함)

- AI 판별여부 체크 (Hugging Face)
  - Wav2Vec2 기반 구조, 음성 특징 추출 후 Classifier 기반 fake/real 평가

- AI 판별 이유 분기별 체크
  - Feature Engineering
  - 음량, 잡음성, 고주파 중심, 스펙트럼 평탄도, 대역폭, 롤오프 기반 평가
    - 음량: 특정 구간의 음성 변화도 체크
    - 잡음성 (ZCR): 특정 구간에서 발화가 크게 튀는지 체크
    - 고주파: 음성이 날카로운지, 답답한지 체크
    - 스펙트럼 평탄도: 음성이 너무 단조롭거나 정돈되어 있는지 체크
    - 대역폭: 음색 분포가 정상적인지 체크
    - 롤오프: 음성이 뚝뚝 끊기는가 체크


## 개선사항
- 결과값 보관 언제까지?
- 초반 build가 오래걸리니, Docker 컨테이너 항시 유지 -> Spring API 호출 방식 채용
- 입력 사운드가 특정 경로에 저장되어야 있어야 함 -> 이 경로를 찾아서 모델이 구동됨.
  - 따라서, 서버 붙이면 FastAPI 업로드 방식으로 변경 필요

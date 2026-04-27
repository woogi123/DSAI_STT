from transformers import pipeline

def detect_phishing_text(full_text):
    # 1. 키워드 기반 매칭
    phishing_keywords = [
        "검찰", "수사관", "경찰", "금융감독원", "안전계좌",
        "송금", "이체", "대출", "정부지원", "본인확인",
        "대포통장", "비밀번호"
    ]

    found_keywords = [word for word in phishing_keywords if word in full_text]

    print("--- [키워드 기반 전처리] ---")
    print(f"의심 단어: {', '.join(found_keywords) if found_keywords else '없음'}")
    print()

    # 2. Hugging Face 기반 텍스트 매칭
    print("--- [보이스피싱 2차 분석] ---")
    print("AI 텍스트 판별 모델 로딩 중...")

    text_model_id = "beomi/KcELECTRA-base-v2022"

    phishing_prob = 0.0
    normal_prob = 0.0

    try:
        text_classifier = pipeline(
            "text-classification",
            model=text_model_id,
            device=0,
            top_k=None
        )

        results = text_classifier(full_text[:512])

        phishing_prob = results[0][0]["score"] * 100
        normal_prob = results[0][1]["score"] * 100

    except Exception:
        print("명확한 근거가 없어 임시조치합니다.")

    print("-" * 50)
    print("🎙️ [보이스피싱 판별 결과]")
    print(f"보이스피싱 의심 확률: {phishing_prob:.2f}%")
    print(f"정상 대화 확률: {normal_prob:.2f}%")
    print("-" * 50)

    return {
        "found_keywords": found_keywords,
        "phishing_prob": phishing_prob,
        "normal_prob": normal_prob
    }


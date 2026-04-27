from transformers import pipeline
import librosa
import numpy as np
import time
import torch

model_id = "garystafford/wav2vec2-deepfake-voice-detector"

print("AI 음성 판별 모델 로딩")
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "audio-classification",
    model=model_id,
    device=device
)

def detect_ai(audio_file_converted):
    audio_file = audio_file_converted
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)

    print(f"[{audio_file}] 조작 여부 분석")
    start_time = time.time()

    chunk_sec = 5
    chunk_size = sr * chunk_sec
    fake_scores = []
    suspicious_sections = []
    threshold = 50.0
    avg_fake = 0.0

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < sr * 2.5:
            continue

        results = classifier({"array": chunk, "sampling_rate": sr}, top_k=None)

        fake_score = None
        real_score = None

        for r in results:
            label = r["label"].lower()
            score = r["score"] * 100

            if "fake" in label or "synthetic" in label:
                fake_score = score
            elif "real" in label or "human" in label:
                real_score = score

        if fake_score is not None:
            fake_scores.append(fake_score)

            start_sec = i / sr
            end_sec = (i + len(chunk)) / sr

            print(f"{start_sec:.1f}s ~ {end_sec:.1f}s -> fake: {fake_score:.2f}%, real: {real_score:.2f}%")

            if fake_score >= threshold:
                suspicious_sections.append((start_sec, end_sec, fake_score))

    print("-" * 50)

    check_AI = 99999

    if fake_scores:
        avg_fake = np.mean(fake_scores)
        print(f"평균 AI 합성 의심 확률: {avg_fake:.2f}%")

        if avg_fake >= 50:
            print("최종 판정: AI 음성 의심")
            check_AI = 1
        else:
            print("최종 판정: 실제 사람 음성 가능성 높음")
            check_AI = 0
    else:
        print("분석 가능한 음성 구간이 부족합니다.")

    print("-" * 50)
    print("🚨 AI 의심 구간")

    if suspicious_sections:
        for start_sec, end_sec, score in suspicious_sections:
            print(f"{start_sec:.1f}s ~ {end_sec:.1f}s 구간: AI 의심 확률 {score:.2f}%")
    else:
        print("의심 구간 없음")

    print("-" * 50)
    print(f"판별 소요 시간: {time.time() - start_time:.2f}초")

    return check_AI, avg_fake, suspicious_sections, audio, sr, chunk_size
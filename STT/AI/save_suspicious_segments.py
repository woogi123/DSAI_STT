# AI 의심 판단 - 구간 분석
#   음량, 잡음성, 고주파 중심, 스펙트럼 평탄도, 대역폭, 롤오프 기반 평가
#   - 음량: 특정 구간의 음성 변화도 체크
#   - 잡음성 (ZCR): 특정 구간에서 발화가 크게 튀는지 체크
#   - 고주파: 음성이 날카로운지, 답답한지 체크
#   - 스펙트럼 평탄도: 음성이 너무 단조롭거나 정돈되어 있는지 체크
#   - 대역폭: 음색 분포가 정상적인지 체크
#   - 롤오프: 음성이 뚝뚝 끊기는가 체크


# input 형식: save_suspicious_segments(check_AI, audio, sr, chunk_size, suspicious_sections)

import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

def save_suspicious_segments(check_AI, audio, sr, chunk_size, suspicious_sections):
    saved_segments = []

    if check_AI != 1:
        return

    print("--- [4단계: 의심 구간 저장 및 근거 확인] ---")

    save_dir = "suspicious_segments"
    os.makedirs(save_dir, exist_ok=True)

    # 정상 구간 기준값 만들기
    normal_sections = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]

        if len(chunk) < sr * 2.5:
            continue

        start_sec = i / sr
        end_sec = (i + len(chunk)) / sr

        is_suspicious = False
        for s, e, _ in suspicious_sections:
            if abs(s - start_sec) < 0.1 and abs(e - end_sec) < 0.1:
                is_suspicious = True
                break

        if not is_suspicious:
            normal_sections.append(chunk)

    # 정상 구간 전체 이어붙이기
    if len(normal_sections) > 0:
        normal_audio = np.concatenate(normal_sections)
    else:
        normal_audio = audio

    # 정상 구간 기준 특징
    base_rms = np.sqrt(np.mean(normal_audio ** 2))
    base_zcr = np.mean(librosa.feature.zero_crossing_rate(y=normal_audio))
    base_cent = np.mean(librosa.feature.spectral_centroid(y=normal_audio, sr=sr))
    base_flat = np.mean(librosa.feature.spectral_flatness(y=normal_audio))
    base_bw = np.mean(librosa.feature.spectral_bandwidth(y=normal_audio, sr=sr))
    base_rolloff = np.mean(librosa.feature.spectral_rolloff(y=normal_audio, sr=sr))

    if suspicious_sections:
        for idx, (start_sec, end_sec, score) in enumerate(suspicious_sections, 1):
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_audio = audio[start_sample:end_sample]

            # 1) wav 저장
            wav_path = os.path.join(save_dir, f"segment_{idx}_{start_sec:.1f}_{end_sec:.1f}.wav")
            sf.write(wav_path, segment_audio, sr)

            # 2) 스펙트로그램 이미지 저장
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(segment_audio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"Suspicious {idx}: {start_sec:.1f}s ~ {end_sec:.1f}s | fake={score:.2f}%")
            plt.tight_layout()

            img_path = os.path.join(save_dir, f"segment_{idx}_{start_sec:.1f}_{end_sec:.1f}.png")
            plt.savefig(img_path)
            plt.close()

            # 3) 의심 구간 특징 계산
            seg_rms = np.sqrt(np.mean(segment_audio ** 2))
            seg_zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment_audio))
            seg_cent = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
            seg_flat = np.mean(librosa.feature.spectral_flatness(y=segment_audio))
            seg_bw = np.mean(librosa.feature.spectral_bandwidth(y=segment_audio, sr=sr))
            seg_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment_audio, sr=sr))

            reasons = []

            if seg_rms > base_rms * 1.5:
                reasons.append("정상 구간보다 음량이 크게 튐")
            elif seg_rms < base_rms * 0.6:
                reasons.append("정상 구간보다 음량이 유독 작음")

            if seg_zcr > base_zcr * 1.4:
                reasons.append("자음·잡음성 성분이 평소보다 많음")
            elif seg_zcr < base_zcr * 0.7:
                reasons.append("발화 패턴이 비정상적으로 단조로움")

            if seg_cent > base_cent * 1.35:
                reasons.append("고주파 중심 에너지가 높음")
            elif seg_cent < base_cent * 0.7:
                reasons.append("고주파 중심 에너지가 낮아 답답하게 들릴 수 있음")

            if seg_flat > base_flat * 1.5:
                reasons.append("스펙트럼이 평탄해 노이즈성/인공적 질감 가능성")
            elif seg_flat < base_flat * 0.65:
                reasons.append("주파수 분포가 유난히 치우쳐 있음")

            if seg_bw > base_bw * 1.35:
                reasons.append("주파수 대역폭이 넓어 평소와 다른 음색 패턴")
            elif seg_bw < base_bw * 0.7:
                reasons.append("주파수 대역폭이 좁아 음색이 비정상적으로 제한됨")

            if seg_rolloff > base_rolloff * 1.3:
                reasons.append("고주파 꼬리 성분이 많음")
            elif seg_rolloff < base_rolloff * 0.75:
                reasons.append("고주파가 일찍 감쇠됨")

            if not reasons:
                reasons.append("정상 구간과 비교해 복합적인 음향 차이는 있으나 단일 특징은 두드러지지 않음")

            print(f"[의심 구간 {idx}] {start_sec:.1f}s ~ {end_sec:.1f}s\n")
            print(f" - AI 의심 확률: {score:.2f}%\n")
            print(f" - 저장된 음성: {wav_path}\n")
            print(f" - 저장된 이미지: {img_path}\n")
            print(f" - 추정 이유: {', '.join(reasons)}\n")
            print("-" * 50)

            saved_segments.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "fake_score": score,
                "saved_wav_path": wav_path,
                "saved_img_path": img_path,
                "reasons": reasons
            })
    else:
        print("의심 구간이 없어 저장할 파일이 없습니다.")
    
    return saved_segments

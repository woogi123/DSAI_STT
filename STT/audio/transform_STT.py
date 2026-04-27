from faster_whisper import WhisperModel
import time

model_size = "large-v3-turbo"

print("모델 로딩 중...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transform_STT(audio_file_converted):
    print(f"[{audio_file_converted}] STT 변환 시작...")
    start_time = time.time()

    segments, info = model.transcribe(audio_file_converted, beam_size=5, language="ko")

    print(f"감지된 언어: {info.language} (확신도: {info.language_probability:.2f})")
    print("-" * 50)

    full_text = ""

    result = full_text.strip()

    print("-" * 50)
    print(f"변환 소요 시간: {time.time() - start_time:.2f}초")
    print("최종 추출 텍스트:\n", result)

    return result


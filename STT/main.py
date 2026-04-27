from audio.transform_wav import transform_wav
from audio.transform_STT import transform_STT
from AI.detect_ai import detect_ai
from AI.save_suspicious_segments import save_suspicious_segments
import json
import sys


def main():
    if len(sys.argv) > 1:
        audio_input = sys.argv[1]

    if audio_input is None:
        print("입력 음성 없음")
        return

    # 1단계: wav 변환
    audio_file_converted = transform_wav(audio_input)
    if audio_file_converted is None:
        print("오디오 변환 실패")
        return

    # 2단계: STT
    full_text = transform_STT(audio_file_converted)

    print("\n" + "=" * 60)
    print("STT 최종 결과")
    print(full_text)
    print("=" * 60 + "\n")

    # 3단계: AI 음성 판별
    result = detect_ai(audio_file_converted)

    if result is None:
        print("AI 음성 판별 실패")
        return

    check_AI, avg_fake, suspicious_sections, audio, sr, chunk_size = result

    # 4단계: 의심 구간 저장 + 저장 결과 받기
    saved_segments = save_suspicious_segments(
        check_AI=check_AI,
        audio=audio,
        sr=sr,
        chunk_size=chunk_size,
        suspicious_sections=suspicious_sections
    )

    if saved_segments is None:
        saved_segments = []

    final_label = "AI 음성 의심" if check_AI == 1 else "실제 사람 음성 가능성 높음"

    json_path = "/app/Audio_result/analysis_result.json"
    txt_path = "/app/Audio_result/analysis_summary.txt"

    # 5단계: 최종 output 생성
    output_data = {
        "input_file": audio_file_converted,
        "stt_text": full_text,
        "ai_probability": avg_fake,
        "final_label": final_label,
        "suspicious_segments": saved_segments
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"입력 파일: {audio_file_converted}\n")
        f.write(f"종합 AI 음성 의심 확률: {avg_fake:.2f}%\n")
        f.write(f"최종 판정: {final_label}\n\n")
        f.write("[STT 결과]\n")
        f.write(full_text + "\n\n")
        f.write("[의심 구간 요약]\n")

        if saved_segments:
            for idx, seg in enumerate(saved_segments, 1):
                f.write(f"{idx}. {seg['start_sec']:.1f}s ~ {seg['end_sec']:.1f}s\n")
                f.write(f"   - AI 의심 확률: {seg['fake_score']:.2f}%\n")
                f.write(f"   - 이유: {', '.join(seg['reasons'])}\n")
                f.write(f"   - 저장 음성: {seg['saved_wav_path']}\n")
                f.write(f"   - 저장 이미지: {seg['saved_img_path']}\n\n")
        else:
            f.write("의심 구간 없음\n")

    print("analysis_result.json 생성 완료")
    print("analysis_summary.txt 생성 완료")


if __name__ == "__main__":
    main()
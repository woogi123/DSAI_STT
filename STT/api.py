from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import json

from audio.transform_wav import transform_wav
from audio.transform_STT import transform_STT
from AI.detect_ai import detect_ai
from AI.save_suspicious_segments import save_suspicious_segments

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    os.makedirs("/app/input", exist_ok=True)
    os.makedirs("/app/Audio_result", exist_ok=True)
    temp_file_path = f"/app/input/{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print("--- [1단계: 오디오 변환] ---")
        audio_file_converted = transform_wav(temp_file_path)
        if audio_file_converted is None:
            return {"error": "오디오 변환 실패"}

        print("--- [2단계: STT 분석] ---")
        full_text = transform_STT(audio_file_converted)

        print("--- [3단계: AI 음성 판별] ---")
        result = detect_ai(audio_file_converted)
        if result is None:
            return {"error": "AI 음성 판별 실패"}

        check_AI, avg_fake, suspicious_sections, audio, sr, chunk_size = result

        print("--- [4단계: 의심 구간 저장] ---")
        saved_segments = save_suspicious_segments(
            check_AI=check_AI, audio=audio, sr=sr,
            chunk_size=chunk_size, suspicious_sections=suspicious_sections
        )
        
        print("--- [5단계: 의심 구간 세분화] ---")
        judgment_basis = []

        if saved_segments:
            # fake_score (AI 의심 확률)가 높은 순서대로 내림차순 정렬
            sorted_segments = sorted(saved_segments, key=lambda x: x.get('fake_score', 0), reverse=True)
            
            for seg in sorted_segments:
                # 5개 까지
                if len(judgment_basis) >= 5:
                    break
                    
                start = seg['start_sec']
                end = seg['end_sec']
                score = seg['fake_score']
                
                if seg.get('reasons'):
                    reasons_str = ", ".join(seg['reasons'])
                    judgment_basis.append(f"🚨 [{start:.1f}s ~ {end:.1f}s] AI 확률 {score:.1f}% : {reasons_str}")

        print("--- [6단계: 결과 출력] ---")
        final_label = "AI 음성 의심" if check_AI == 1 else "실제 사람 음성 가능성 높음"

        output_data = {
            "input_file": audio_file_converted,
            "stt_text": full_text,
            "ai_probability": avg_fake,
            "final_label": final_label,
            "suspicious_segments": saved_segments or [],
            "judgment_basis": judgment_basis
        }

        json_path = "/app/Audio_result/analysis_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_data

    except Exception as e:
        return {"error": f"서버 내부 오류: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
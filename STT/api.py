from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil
import os
import json
import base64
import uuid

# 기존 모듈 임포트
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

os.makedirs("/app/input", exist_ok=True)
os.makedirs("/app/Audio_result", exist_ok=True)

def process_audio_file(audio_path: str):
    print("--- [1단계: 오디오 변환] ---")
    audio_file_converted = transform_wav(audio_path)
    if audio_file_converted is None:
        raise HTTPException(status_code=500, detail="오디오 변환 실패")

    print("--- [2단계: STT 분석] ---")
    full_text = transform_STT(audio_file_converted)

    print("--- [3단계: AI 음성 판별] ---")
    result = detect_ai(audio_file_converted)
    if result is None:
        raise HTTPException(status_code=500, detail="AI 음성 판별 실패")

    check_AI, avg_fake, suspicious_sections, audio, sr, chunk_size = result

    print("--- [4단계: 의심 구간 저장] ---")
    saved_segments = save_suspicious_segments(
        check_AI=check_AI, audio=audio, sr=sr,
        chunk_size=chunk_size, suspicious_sections=suspicious_sections
    )
    
    print("--- [5단계: 의심 구간 세분화 ---")
    judgment_basis = []
    if saved_segments:
        # fake_score (AI 의심 확률)가 높은 순서대로 내림차순 정렬
        sorted_segments = sorted(saved_segments, key=lambda x: x.get('fake_score', 0), reverse=True)
        for seg in sorted_segments:
            if len(judgment_basis) >= 5:
                break
            start = seg['start_sec']
            end = seg['end_sec']
            score = seg['fake_score']
            if seg.get('reasons'):
                reasons_str = ", ".join(seg['reasons'])
                judgment_basis.append(f"🚨 [{start:.1f}s ~ {end:.1f}s] AI 확률 {score:.1f}% : {reasons_str}")

    final_label = "AI 음성 의심" if check_AI == 1 else "실제 사람 음성 가능성 높음"
    if not judgment_basis:
        judgment_basis.append(final_label)

    # 로컬 JSON 기록용
    output_data = {
        "input_file": audio_file_converted,
        "stt_text": full_text,
        "ai_probability": avg_fake,
        "final_label": final_label,
        "suspicious_segments": saved_segments or [],
        "judgment_basis": judgment_basis
    }
    json_path = f"/app/Audio_result/analysis_result_{uuid.uuid4().hex[:8]}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 💡 Next.js 프론트엔드가 요구하는 포맷으로 리턴
    return {
        "manipulationScore": int(avg_fake),
        "reasons": judgment_basis, # 우기님이 만든 예쁜 문자열을 화면에 출력!
        "stt_text": full_text
    }


# =====================================================================
# API 1: 웹(Next.js)에서 파일 직접 업로드 시 사용하는 주소
# =====================================================================
@app.post("/api/analyze-audio")
async def analyze_audio_upload_api(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="업로드된 파일이 없습니다.")

    temp_input_path = f"/app/input/{uuid.uuid4().hex}_{file.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result_data = process_audio_file(temp_input_path)
        return JSONResponse(content=result_data)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


# =====================================================================
# API 2: Visual 탐지기에서 Base64로 추출된 오디오를 쏠 때 사용하는 주소
# =====================================================================
class AudioDataPayload(BaseModel):
    format: str
    encoding: str
    data: str

@app.post("/api/audio-extract-file")
async def analyze_extracted_audio_api(payload: AudioDataPayload):
    if payload.encoding != "base64" or not payload.data:
        raise HTTPException(status_code=400, detail="올바른 Base64 데이터가 필요합니다.")

    temp_input_path = f"/app/input/extracted_{uuid.uuid4().hex}.{payload.format}"

    try:
        audio_bytes = base64.b64decode(payload.data)
        with open(temp_input_path, "wb") as f:
            f.write(audio_bytes)

        result_data = process_audio_file(temp_input_path)
        return JSONResponse(content={"status": "success", **result_data})
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

# 실행 및 에러 시 처리
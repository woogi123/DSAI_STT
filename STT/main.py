from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
import json
import base64
import uuid

# 기존 모듈 임포트 (수정 없이 그대로 사용)
from audio.transform_wav import transform_wav
from audio.transform_STT import transform_STT
from AI.detect_ai import detect_ai
from AI.save_suspicious_segments import save_suspicious_segments

app = FastAPI()

# 컨테이너 안에서 결과물을 저장할 기본 경로 설정
RESULT_DIR = "/app/Audio_result"
os.makedirs(RESULT_DIR, exist_ok=True)

# 💡 [핵심] 핵심 로직을 묶어둔 공통 함수 (어떤 방식으로 받든 처리는 동일함)
def process_audio_file(audio_path: str):
    """
    저장된 음성 파일을 받아 STT 및 AI 판별을 수행하는 공통 로직
    """
    # 1단계: wav 변환
    audio_file_converted = transform_wav(audio_path)
    if audio_file_converted is None:
        raise HTTPException(status_code=500, detail="오디오 변환 실패")

    # 2단계: STT 변환
    full_text = transform_STT(audio_file_converted)

    # 3단계: AI 음성 판별
    result = detect_ai(audio_file_converted)
    if result is None:
        raise HTTPException(status_code=500, detail="AI 음성 판별 실패")

    check_AI, avg_fake, suspicious_sections, audio, sr, chunk_size = result

    # 4단계: 의심 구간 저장
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

    # 기록용 JSON 파일 저장
    json_path = os.path.join(RESULT_DIR, f"result_{uuid.uuid4().hex[:8]}.json")
    output_data = {
        "input_file": audio_file_converted,
        "stt_text": full_text,
        "ai_probability": avg_fake,
        "final_label": final_label,
        "suspicious_segments": saved_segments
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 프론트엔드 응답용 포맷팅
    reasons_list = []
    if check_AI == 1 and saved_segments:
        for seg in saved_segments:
            reasons_list.append(f"{seg['start_sec']:.1f}s~{seg['end_sec']:.1f}s 구간: {', '.join(seg['reasons'])}")
    else:
        reasons_list.append(final_label)

    return {
        "manipulationScore": int(avg_fake),
        "reasons": reasons_list,
        "stt_text": full_text
    }


# =====================================================================
# API 1: 홈페이지(Next.js)에서 음성 파일을 직접 업로드 했을 때
# =====================================================================
@app.post("/analyze-audio")
async def analyze_audio_upload_api(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="업로드된 파일이 없습니다.")

    # 임시 저장소에 파일 저장
    temp_filename = f"upload_{uuid.uuid4().hex}_{file.filename}"
    temp_input_path = f"/tmp/{temp_filename}"
    
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 공통 처리 로직 호출
        result_data = process_audio_file(temp_input_path)
        return JSONResponse(content=result_data)
    except Exception as e:
        print(f"Error processing uploaded audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)


# =====================================================================
# API 2: Visual 탐지기에서 추출한 Base64 오디오 데이터를 넘겨받을 때
# =====================================================================
class AudioDataPayload(BaseModel):
    format: str
    encoding: str
    data: str

@app.post("/api/v1/analyze-extracted-audio")
async def analyze_extracted_audio_api(payload: AudioDataPayload):
    if payload.encoding != "base64" or not payload.data:
        raise HTTPException(status_code=400, detail="올바른 Base64 오디오 데이터가 필요합니다.")

    # 임시 저장소에 파일 생성
    temp_filename = f"extracted_{uuid.uuid4().hex}.{payload.format}"
    temp_input_path = f"/tmp/{temp_filename}"

    try:
        # Base64를 디코딩해서 파일로 만들기
        audio_bytes = base64.b64decode(payload.data)
        with open(temp_input_path, "wb") as f:
            f.write(audio_bytes)

        # 공통 처리 로직 호출 (API 1과 완벽히 동일한 결과 도출!)
        result_data = process_audio_file(temp_input_path)
        
        # 외부 연동용이므로 성공 여부(status)를 추가해서 반환
        return JSONResponse(content={
            "status": "success",
            **result_data
        })
    except Exception as e:
        print(f"Error processing extracted audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#!pip install faster-whisper

import os
import subprocess

def transform_wav(audio_input):
    input_file = audio_input
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    audio_file_converted = "input.wav"

    print("--- [1단계: 오디오 파일 확인 및 변환] ---")
    
    if not os.path.exists(input_file):
        print(f"에러: '{input_file}' 파일이 없습니다.")
        return None
    
    print(f"'{input_file}' 파일을 표준 포맷(.wav)으로 변환합니다...")
    
    subprocess.run(
        ['ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', audio_file_converted, '-y'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    print("변환 완료")
    return audio_file_converted
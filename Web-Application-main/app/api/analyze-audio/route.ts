import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 300;

// interface SuspiciousSegment {
//   start_sec: number;
//   end_sec: number;
//   fake_score: number;
// }

export async function POST(req: Request) {
  const form = await req.formData();
  const file = form.get("file");
  
  if (!(file instanceof File)) {
    return NextResponse.json({ error: "file이 필요합니다." }, { status: 400 });
  }

  try {
    const sttResponse = await fetch("http://stt-ai:8000/api/analyze-audio", {
      method: "POST",
      body: form,
    });

    if (!sttResponse.ok) {
      const errorText = await sttResponse.text();
      return NextResponse.json(
        { error: `STT 서버 오류: ${errorText}` },
        { status: sttResponse.status }
      );
    }

    const data = await sttResponse.json();

    if (data.error) {
      return NextResponse.json({ error: data.error }, { status: 500 });
    }

    const manipulationScore = data.manipulationScore || 0;
    const reasons = data.reasons || ["특이사항 없음"];
    const sttText = data.stt_text || "음성 텍스트를 추출하지 못했습니다.";

    return NextResponse.json({ 
        manipulationScore, 
        reasons,
        sttText 
    });


  } catch (error: unknown) {
    console.error("통신 에러:", error);
    return NextResponse.json(
      { error: "STT 컨테이너와 통신할 수 없습니다. 도커가 켜져 있는지 확인하세요." }, 
      { status: 500 }
    );
  }
}
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
    const sttResponse = await fetch("http://stt-ai:8000/analyze", {
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

    const manipulationScore = Math.round(data.ai_probability || 0);
    
    let reasons: string[] = [];
    if (data.judgment_basis && data.judgment_basis.length > 0) {
      reasons = data.judgment_basis;
    } else {
      reasons.push(data.final_label || "특이사항 없음");
    }

    return NextResponse.json({ manipulationScore, reasons });

  } catch (error: unknown) {
    console.error("통신 에러:", error);
    return NextResponse.json(
      { error: "STT 컨테이너와 통신할 수 없습니다. 도커가 켜져 있는지 확인하세요." }, 
      { status: 500 }
    );
  }
}
import { NextResponse } from "next/server";

export async function GET() {
  const baseUrl = process.env.CLINICIAN_TOOLS_URL;

  if (!baseUrl) {
    return NextResponse.json(
      { status: "unconfigured", service: "clinician-tools" },
      { status: 503 },
    );
  }

  try {
    const response = await fetch(new URL("/health", baseUrl), {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { status: "unavailable", service: "clinician-tools" },
        { status: 502 },
      );
    }

    const health = await response.json();
    return NextResponse.json({
      status: "connected",
      service: health.service,
      backend: health.status,
    });
  } catch {
    return NextResponse.json(
      { status: "unavailable", service: "clinician-tools" },
      { status: 502 },
    );
  }
}

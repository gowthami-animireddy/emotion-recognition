import { type NextRequest, NextResponse } from "next/server"
import type { BatchPredictionRequest, BatchPredictionResponse } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const body: BatchPredictionRequest = await request.json()

    if (!body.samples || body.samples.length === 0) {
      return NextResponse.json({ error: "No samples provided" }, { status: 400 })
    }

    // Process each sample through the predict endpoint
    const results = await Promise.all(
      body.samples.map(async (sample) => {
        const response = await fetch(new URL("/api/predict", request.url), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sample),
        })
        return response.json()
      }),
    )

    const batchResponse: BatchPredictionResponse = {
      status: "success",
      data: {
        total_samples: body.samples.length,
        successful: results.filter((r) => r.status === "success").length,
        failed: results.filter((r) => r.status !== "success").length,
        results: results,
      },
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json(batchResponse)
  } catch (error) {
    console.error("[v0] Batch prediction error:", error)
    return NextResponse.json({ error: "Failed to process batch prediction" }, { status: 500 })
  }
}

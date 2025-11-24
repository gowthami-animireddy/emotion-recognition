import type { PredictionRequest, PredictionResponse, BatchPredictionRequest } from "./types"

export async function predictEmotion(input: PredictionRequest): Promise<PredictionResponse> {
  try {
    console.log("[v0] Calling /api/predict with input:", input)

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    })

    console.log("[v0] Response status:", response.status)

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      console.error("[v0] API error:", errorData)
      throw new Error(errorData.error || `HTTP ${response.status}: Failed to get emotion prediction`)
    }

    const data = await response.json()
    console.log("[v0] Prediction result:", data)
    return data
  } catch (error) {
    console.error("[v0] Fetch error in predictEmotion:", error)
    throw error
  }
}

export async function batchPredict(request: BatchPredictionRequest) {
  try {
    const response = await fetch("/api/batch-predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.error || `HTTP ${response.status}: Failed to get batch predictions`)
    }

    return response.json()
  } catch (error) {
    console.error("[v0] Fetch error in batchPredict:", error)
    throw error
  }
}

export async function checkHealth() {
  try {
    const response = await fetch("/api/health")
    return response.ok
  } catch (error) {
    console.error("[v0] Health check failed:", error)
    return false
  }
}

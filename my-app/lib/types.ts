export interface EmotionPrediction {
  primary_emotion: string
  confidence: number
  all_emotions: Record<string, number>
  attention_weights: {
    text: number
    audio: number
    video: number
  }
}

export interface PredictionRequest {
  text?: string
  audio_features?: number[]
  video_features?: number[]
}

export interface PredictionResponse {
  status: "success" | "error"
  data?: EmotionPrediction
  error?: string
  timestamp: string
}

export interface BatchPredictionRequest {
  samples: PredictionRequest[]
}

export interface BatchPredictionResponse {
  status: "success" | "error"
  data?: {
    total_samples: number
    successful: number
    failed: number
    results: PredictionResponse[]
  }
  error?: string
  timestamp: string
}

export type Emotion = "angry" | "disgust" | "fear" | "happy" | "neutral" | "sad" | "surprise"

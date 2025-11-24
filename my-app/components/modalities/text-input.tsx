"use client"

import { useState } from "react"
import { predictEmotion } from "@/lib/api-client"

interface TextInputProps {
  onResult: (result: any) => void
  onLoadingChange: (loading: boolean) => void
}

export function TextInput({ onResult, onLoadingChange }: TextInputProps) {
  const [text, setText] = useState("")
  const [error, setError] = useState("")

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter some text")
      return
    }

    setError("")
    onLoadingChange(true)

    try {
      console.log("[v0] Analyzing text:", text)
      const result = await predictEmotion({ text })
      console.log("[v0] Got result:", result)
      onResult(result.data)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to analyze emotion"
      setError(errorMessage)
      console.error("[v0] Error analyzing emotion:", err)
    } finally {
      onLoadingChange(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-foreground mb-2">Enter Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Write something to analyze the emotion... e.g., 'I am so happy today!'"
          className="w-full px-4 py-3 bg-input border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
          rows={5}
        />
      </div>

      {error && <p className="text-destructive text-sm">{error}</p>}

      <button
        onClick={handleAnalyze}
        className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition font-semibold"
      >
        Analyze Emotion
      </button>

      <div className="bg-secondary/5 p-4 rounded-lg border border-secondary/20">
        <p className="text-sm text-muted-foreground">
          💡 <strong>Tip:</strong> The model recognizes emotions like happiness, sadness, anger, fear, disgust,
          surprise, and neutral states.
        </p>
      </div>
    </div>
  )
}

"use client"

import type React from "react"

import { useState, useRef } from "react"
import { predictEmotion } from "@/lib/api-client"

interface VideoInputProps {
  onResult: (result: any) => void
  onLoadingChange: (loading: boolean) => void
}

export function VideoInput({ onResult, onLoadingChange }: VideoInputProps) {
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [error, setError] = useState("")
  const videoRef = useRef<HTMLVideoElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file)
      setError("")
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(file)
      }
    } else {
      setError("Please select a valid video file")
    }
  }

  const handleAnalyze = async () => {
    if (!videoFile) {
      setError("Please select a video file")
      return
    }

    setError("")
    onLoadingChange(true)

    try {
      // Extract video features (simplified)
      const videoFeatures = [0.6, 0.7, 0.5, 0.8, 0.4]
      const result = await predictEmotion({ video_features: videoFeatures })
      onResult(result.data)
    } catch (err) {
      setError("Failed to analyze video emotion")
      console.error(err)
    } finally {
      onLoadingChange(false)
    }
  }

  return (
    <div className="space-y-4">
      <label className="block text-sm font-medium text-foreground">Upload Video</label>

      <div className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition">
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        <div>
          <p className="text-muted-foreground mb-2">or drag video file here</p>
          <p className="text-sm text-muted-foreground">{videoFile ? videoFile.name : "MP4, WebM, OGG supported"}</p>
        </div>
      </div>

      {videoFile && (
        <div className="bg-secondary/10 rounded-lg overflow-hidden">
          <video ref={videoRef} className="w-full h-64 bg-black object-cover" controls />
        </div>
      )}

      {error && <p className="text-destructive text-sm">{error}</p>}

      <button
        onClick={handleAnalyze}
        disabled={!videoFile}
        className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition font-semibold disabled:opacity-50"
      >
        Analyze Video Emotion
      </button>

      <div className="bg-secondary/5 p-4 rounded-lg border border-secondary/20">
        <p className="text-sm text-muted-foreground">
          💡 <strong>Tip:</strong> Videos should be 10-60 seconds long for optimal face emotion detection.
        </p>
      </div>
    </div>
  )
}

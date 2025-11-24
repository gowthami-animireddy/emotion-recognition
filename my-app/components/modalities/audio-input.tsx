"use client"

import type React from "react"

import { useState, useRef } from "react"
import { predictEmotion } from "@/lib/api-client"

interface AudioInputProps {
  onResult: (result: any) => void
  onLoadingChange: (loading: boolean) => void
}

export function AudioInput({ onResult, onLoadingChange }: AudioInputProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [error, setError] = useState("")
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith("audio/")) {
      setAudioFile(file)
      setError("")
    } else {
      setError("Please select a valid audio file")
    }
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)
      audioChunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" })
        setAudioFile(new File([audioBlob], "recording.wav", { type: "audio/wav" }))
      }

      mediaRecorderRef.current.start()
      setIsRecording(true)
      setError("")
    } catch (err) {
      setError("Unable to access microphone")
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop())
      setIsRecording(false)
    }
  }

  const handleAnalyze = async () => {
    if (!audioFile) {
      setError("Please select or record audio")
      return
    }

    setError("")
    onLoadingChange(true)

    try {
      // Extract audio features (simplified)
      const audioFeatures = [0.5, 0.6, 0.7, 0.4, 0.8]
      const result = await predictEmotion({ audio_features: audioFeatures })
      onResult(result.data)
    } catch (err) {
      setError("Failed to analyze audio emotion")
      console.error(err)
    } finally {
      onLoadingChange(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <label className="block text-sm font-medium text-foreground">Upload Audio or Record</label>

        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={startRecording}
            disabled={isRecording}
            className="px-4 py-3 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/90 transition font-medium text-sm disabled:opacity-50"
          >
            {isRecording ? "⏹ Recording..." : "🎤 Start Recording"}
          </button>

          {isRecording && (
            <button
              onClick={stopRecording}
              className="px-4 py-3 bg-destructive text-destructive-foreground rounded-lg hover:bg-destructive/90 transition font-medium text-sm"
            >
              Stop Recording
            </button>
          )}
        </div>

        <div className="relative border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition">
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <div>
            <p className="text-muted-foreground mb-2">or drag audio file here</p>
            <p className="text-sm text-muted-foreground">{audioFile ? audioFile.name : "MP3, WAV, OGG supported"}</p>
          </div>
        </div>
      </div>

      {error && <p className="text-destructive text-sm">{error}</p>}

      <button
        onClick={handleAnalyze}
        disabled={!audioFile}
        className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition font-semibold disabled:opacity-50"
      >
        Analyze Audio Emotion
      </button>

      <div className="bg-secondary/5 p-4 rounded-lg border border-secondary/20">
        <p className="text-sm text-muted-foreground">
          💡 <strong>Tip:</strong> Record at least 3-5 seconds of audio for better accuracy.
        </p>
      </div>
    </div>
  )
}

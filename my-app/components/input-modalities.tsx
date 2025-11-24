"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Zap, Volume2, Video } from "lucide-react"

interface InputModalitiesProps {
  onAnalyze: (data: any) => void
  isLoading: boolean
}

export function InputModalities({ onAnalyze, isLoading }: InputModalitiesProps) {
  const [text, setText] = useState("")
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState("text")

  const handleTextSubmit = async () => {
    if (text.trim()) {
      onAnalyze({
        text: text,
        input_type: "text",
      })
      setText("")
    }
  }

  const handleAudioSubmit = async () => {
    if (audioFile) {
      const formData = new FormData()
      formData.append("audio", audioFile)

      try {
        const response = await fetch("/api/analyze-audio", {
          method: "POST",
          body: formData,
        })
        const result = await response.json()
        onAnalyze({
          audio_file: audioFile,
          input_type: "audio",
          prediction: result,
        })
        setAudioFile(null)
      } catch (error) {
        console.error("Audio analysis failed:", error)
      }
    }
  }

  const handleVideoSubmit = async () => {
    if (videoFile) {
      const formData = new FormData()
      formData.append("video", videoFile)

      try {
        const response = await fetch("/api/analyze-video", {
          method: "POST",
          body: formData,
        })
        const result = await response.json()
        onAnalyze({
          video_file: videoFile,
          input_type: "video",
          prediction: result,
        })
        setVideoFile(null)
      } catch (error) {
        console.error("Video analysis failed:", error)
      }
    }
  }

  return (
    <div className="p-6 space-y-4 flex-1 flex flex-col">
      {/* Tab Navigation */}
      <div className="flex gap-2 bg-gradient-to-r from-primary/10 to-accent/10 rounded-lg p-1 backdrop-blur-sm border border-primary/20">
        {[
          { id: "text", label: "Text", icon: "📝" },
          { id: "audio", label: "Audio", icon: "🎵" },
          { id: "video", label: "Video", icon: "🎬" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all flex items-center justify-center gap-2 ${
              activeTab === tab.id
                ? "bg-gradient-to-r from-primary to-secondary text-white shadow-lg shadow-primary/40"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <span>{tab.icon}</span>
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="flex-1 flex flex-col">
        {activeTab === "text" && (
          <div className="space-y-3 flex flex-col h-full">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to analyze your emotional tone..."
              className="h-32 flex-1 bg-gradient-to-br from-card/50 to-card/30 border-primary/30 focus:border-primary/70 resize-none"
            />
            <Button
              onClick={handleTextSubmit}
              disabled={isLoading || !text.trim()}
              className="w-full bg-gradient-to-r from-primary via-secondary to-accent hover:shadow-lg hover:shadow-primary/50 transition-all text-white font-semibold"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin mr-2 h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Analyze Text
                </>
              )}
            </Button>
          </div>
        )}

        {activeTab === "audio" && (
          <div className="space-y-3 flex flex-col h-full justify-between">
            <label className="flex-1 flex items-center justify-center gap-3 p-6 border-2 border-dashed border-secondary/50 rounded-lg cursor-pointer hover:border-secondary hover:bg-secondary/5 transition-all group bg-gradient-to-br from-secondary/5 to-transparent">
              <Volume2 className="w-6 h-6 text-secondary group-hover:text-secondary/80 transition" />
              <div className="text-center">
                <p className="text-sm font-medium">{audioFile ? audioFile.name : "Upload audio"}</p>
                <p className="text-xs text-muted-foreground">.mp3, .wav, .m4a</p>
              </div>
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
              />
            </label>
            {audioFile && (
              <Button
                onClick={handleAudioSubmit}
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-secondary to-secondary/70 hover:shadow-lg hover:shadow-secondary/50 transition-all text-white font-semibold"
              >
                {isLoading ? "Analyzing..." : "Analyze Audio"}
              </Button>
            )}
          </div>
        )}

        {activeTab === "video" && (
          <div className="space-y-3 flex flex-col h-full justify-between">
            <label className="flex-1 flex items-center justify-center gap-3 p-6 border-2 border-dashed border-accent/50 rounded-lg cursor-pointer hover:border-accent hover:bg-accent/5 transition-all group bg-gradient-to-br from-accent/5 to-transparent">
              <Video className="w-6 h-6 text-accent group-hover:text-accent/80 transition" />
              <div className="text-center">
                <p className="text-sm font-medium">{videoFile ? videoFile.name : "Upload video"}</p>
                <p className="text-xs text-muted-foreground">.mp4, .webm, .mov</p>
              </div>
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
              />
            </label>
            {videoFile && (
              <Button
                onClick={handleVideoSubmit}
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-accent to-accent/70 hover:shadow-lg hover:shadow-accent/50 transition-all text-white font-semibold"
              >
                {isLoading ? "Analyzing..." : "Analyze Video"}
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

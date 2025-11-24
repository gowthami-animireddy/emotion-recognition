"use client"

import { useState } from "react"
import { TextInput } from "./modalities/text-input"
import { AudioInput } from "./modalities/audio-input"
import { VideoInput } from "./modalities/video-input"
import { ResultsPanel } from "./results-panel"

export function ModalitySelector() {
  const [activeTab, setActiveTab] = useState<"text" | "audio" | "video">("text")
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)

  const tabs = [
    { id: "text", label: "Text Analysis", icon: "📝" },
    { id: "audio", label: "Audio Recognition", icon: "🎤" },
    { id: "video", label: "Video Detection", icon: "📹" },
  ]

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-1">
        <div className="bg-card border border-border rounded-lg p-4 sticky top-24">
          <h3 className="font-semibold text-foreground mb-4">Select Modality</h3>
          <div className="space-y-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full px-4 py-3 rounded-lg font-medium transition text-left ${
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary/10 text-foreground hover:bg-secondary/20"
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="lg:col-span-2 space-y-6">
        <div className="bg-card border border-border rounded-xl p-8">
          {activeTab === "text" && <TextInput onResult={setResults} onLoadingChange={setLoading} />}
          {activeTab === "audio" && <AudioInput onResult={setResults} onLoadingChange={setLoading} />}
          {activeTab === "video" && <VideoInput onResult={setResults} onLoadingChange={setLoading} />}
        </div>

        {results && !loading && <ResultsPanel results={results} />}
      </div>
    </div>
  )
}

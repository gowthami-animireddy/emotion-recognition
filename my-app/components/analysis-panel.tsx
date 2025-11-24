"use client"

import { format } from "date-fns"
import { TrendingUp } from "lucide-react"

interface AnalysisPanelProps {
  analysis: any
}

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  const emotionColors: Record<string, { gradient: string; icon: string }> = {
    angry: { gradient: "from-red-500 to-rose-500", icon: "🔥" },
    anxious: { gradient: "from-amber-500 to-orange-500", icon: "😰" },
    contempt: { gradient: "from-purple-500 to-violet-500", icon: "😒" },
    content: { gradient: "from-emerald-500 to-green-500", icon: "😌" },
    disgusted: { gradient: "from-lime-500 to-green-500", icon: "🤢" },
    excited: { gradient: "from-pink-500 to-rose-500", icon: "🎉" },
    fear: { gradient: "from-cyan-500 to-blue-500", icon: "😨" },
    sad: { gradient: "from-indigo-500 to-blue-500", icon: "😢" },
    surprised: { gradient: "from-orange-500 to-amber-500", icon: "😮" },
    neutral: { gradient: "from-slate-500 to-gray-500", icon: "😐" },
  }

  const emotionData = emotionColors[analysis.primary_emotion] || {
    gradient: "from-primary to-secondary",
    icon: "❓",
  }

  return (
    <div className="bg-gradient-to-br from-card to-card/50 border border-border/50 rounded-xl p-8 backdrop-blur-sm shadow-lg hover:shadow-xl transition-shadow">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left: Main Emotion Display */}
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-3">Primary Emotion</p>
          <div className="mb-6">
            <div className="flex items-center gap-4 mb-4">
              <span className="text-6xl">{emotionData.icon}</span>
              <h2
                className={`text-5xl font-bold bg-gradient-to-r ${emotionData.gradient} bg-clip-text text-transparent`}
              >
                {analysis.primary_emotion.charAt(0).toUpperCase() + analysis.primary_emotion.slice(1)}
              </h2>
            </div>
            <p className="text-sm text-muted-foreground">{format(new Date(analysis.timestamp), "PPpp")}</p>
          </div>
        </div>

        {/* Right: Confidence Score */}
        <div className="flex flex-col items-end justify-center">
          <p className="text-sm font-medium text-muted-foreground mb-2">Confidence Score</p>
          <div className="text-right">
            <div
              className={`text-6xl font-bold bg-gradient-to-r ${emotionData.gradient} bg-clip-text text-transparent mb-2`}
            >
              {(analysis.confidence * 100).toFixed(1)}%
            </div>
            <div
              className={`flex items-center justify-end gap-2 text-transparent bg-gradient-to-r ${emotionData.gradient} bg-clip-text`}
            >
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm font-medium">Confidence</span>
            </div>
          </div>
        </div>
      </div>

      {/* Emotion Distribution */}
      <div className="mt-8 pt-8 border-t border-border/30">
        <p className="text-sm font-medium text-muted-foreground mb-4">Emotion Distribution</p>
        <div className="space-y-3">
          {Object.entries(analysis.all_emotions)
            .sort(([, a]: any, [, b]: any) => b - a)
            .map(([emotion, score]: any) => {
              const emotionColor = emotionColors[emotion]
              return (
                <div key={emotion} className="group">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm font-medium capitalize text-foreground/80 group-hover:text-foreground transition">
                      {emotionColor?.icon} {emotion}
                    </span>
                    <span className="text-sm font-semibold text-primary">{(score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2.5 bg-muted/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${emotionColor?.gradient || "from-primary to-secondary"} rounded-full transition-all duration-500 group-hover:shadow-lg group-hover:shadow-primary/50`}
                      style={{
                        width: `${score * 100}%`,
                      }}
                    />
                  </div>
                </div>
              )
            })}
        </div>
      </div>
    </div>
  )
}

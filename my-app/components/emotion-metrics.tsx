"use client"

import { Card } from "@/components/ui/card"

export function EmotionMetrics({ analysis }: { analysis: any }) {
  const modalities = [
    {
      name: "Text",
      value: analysis.attention_weights.text,
      icon: "📝",
      color: "from-indigo-500 to-blue-500",
      shadowColor: "shadow-indigo-500/50",
    },
    {
      name: "Audio",
      value: analysis.attention_weights.audio,
      icon: "🎵",
      color: "from-pink-500 to-rose-500",
      shadowColor: "shadow-pink-500/50",
    },
    {
      name: "Video",
      value: analysis.attention_weights.video,
      icon: "🎬",
      color: "from-emerald-500 to-green-500",
      shadowColor: "shadow-emerald-500/50",
    },
  ]

  return (
    <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 p-6 backdrop-blur-sm">
      <div className="mb-6">
        <h3 className="text-lg font-semibold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          Modality Contributions
        </h3>
        <p className="text-sm text-muted-foreground">How each input type influenced the result</p>
      </div>
      <div className="space-y-5">
        {modalities.map((mod) => (
          <div key={mod.name} className="group">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <span className="text-xl">{mod.icon}</span>
                <span className="text-sm font-semibold">{mod.name}</span>
              </div>
              <span className={`text-sm font-bold text-transparent bg-gradient-to-r ${mod.color} bg-clip-text`}>
                {(mod.value * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-3 bg-muted/50 rounded-full overflow-hidden">
              <div
                className={`h-full bg-gradient-to-r ${mod.color} rounded-full transition-all duration-500 group-hover:shadow-lg group-hover:${mod.shadowColor}`}
                style={{
                  width: `${mod.value * 100}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Insights */}
      <div className="mt-6 pt-6 border-t border-border/30">
        <p className="text-xs font-medium text-muted-foreground mb-2">INSIGHTS</p>
        <p className="text-sm text-foreground/70">
          {analysis.attention_weights.text > 0.5
            ? "Text analysis is the primary emotion indicator for this input."
            : "Multi-modal inputs are influencing this emotion detection."}
        </p>
      </div>
    </Card>
  )
}

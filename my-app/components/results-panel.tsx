"use client"

interface ResultsPanelProps {
  results: {
    primary_emotion: string
    confidence: number
    all_emotions: Record<string, number>
    attention_weights: {
      text: number
      audio: number
      video: number
    }
  }
}

export function ResultsPanel({ results }: ResultsPanelProps) {
  const emotionIcons: Record<string, string> = {
    angry: "😠",
    disgust: "🤢",
    fear: "😨",
    happy: "😊",
    neutral: "😐",
    sad: "😢",
    surprise: "😲",
  }

  return (
    <div className="bg-gradient-to-br from-primary/10 to-accent/10 border border-primary/20 rounded-xl p-8 space-y-6">
      <div className="text-center">
        <div className="text-6xl mb-4">{emotionIcons[results.primary_emotion] || "🤔"}</div>
        <h3 className="text-3xl font-bold text-foreground capitalize">{results.primary_emotion}</h3>
        <p className="text-xl text-muted-foreground mt-2">Confidence: {(results.confidence * 100).toFixed(1)}%</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {Object.entries(results.all_emotions).map(([emotion, score]) => (
          <div key={emotion} className="bg-card border border-border rounded-lg p-3">
            <p className="text-sm font-medium text-foreground capitalize mb-1">{emotion}</p>
            <div className="w-full bg-secondary/20 rounded-full h-2 overflow-hidden">
              <div className="bg-primary h-full transition-all duration-500" style={{ width: `${score * 100}%` }} />
            </div>
            <p className="text-xs text-muted-foreground mt-1">{(score * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>

      <div className="bg-card border border-border rounded-lg p-4">
        <h4 className="font-semibold text-foreground mb-4">Modality Contribution</h4>
        <div className="space-y-3">
          {[
            { name: "Text", value: results.attention_weights.text, icon: "📝" },
            { name: "Audio", value: results.attention_weights.audio, icon: "🎤" },
            { name: "Video", value: results.attention_weights.video, icon: "📹" },
          ].map(({ name, value, icon }) => (
            <div key={name} className="flex items-center gap-3">
              <span className="text-lg">{icon}</span>
              <span className="text-sm font-medium text-foreground flex-1">{name}</span>
              <div className="w-32 bg-secondary/20 rounded-full h-2 overflow-hidden">
                <div className="bg-secondary h-full transition-all duration-500" style={{ width: `${value * 100}%` }} />
              </div>
              <span className="text-xs text-muted-foreground w-12 text-right">{(value * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

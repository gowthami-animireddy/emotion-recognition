"use client"

import { Card } from "@/components/ui/card"
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from "recharts"

export function EmotionDistributionChart({ emotions }: { emotions: Record<string, number> }) {
  const data = Object.entries(emotions).map(([emotion, score]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    value: Math.round(score * 100),
    fullMark: 100,
  }))

  const emotionColors: Record<string, string> = {
    angry: "#ef4444",
    anxious: "#f59e0b",
    contempt: "#a855f7",
    content: "#10b981",
    disgusted: "#84cc16",
    excited: "#ec4899",
    fear: "#06b6d4",
    sad: "#4f46e5",
    surprised: "#f97316",
    neutral: "#64748b",
  }

  return (
    <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 p-6 backdrop-blur-sm">
      <h3 className="text-lg font-semibold mb-4">Emotion Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="emotion" tick={{ fontSize: 12 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} />
          <Radar name="Emotion Score" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
        </RadarChart>
      </ResponsiveContainer>
    </Card>
  )
}

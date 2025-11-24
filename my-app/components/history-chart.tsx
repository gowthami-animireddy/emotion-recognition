"use client"

import { Card } from "@/components/ui/card"
import { BarChart } from "lucide-react"

export function HistoryChart({ analyses }: { analyses: any[] }) {
  const emotionCounts = analyses.slice(0, 20).reduce(
    (acc, a) => {
      acc[a.primary_emotion] = (acc[a.primary_emotion] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  const totalAnalyses = Object.values(emotionCounts).reduce((a, b) => a + b, 0)

  const emotionColors: Record<string, { bg: string; shadow: string }> = {
    angry: { bg: "bg-red-500", shadow: "shadow-red-500/50" },
    anxious: { bg: "bg-amber-500", shadow: "shadow-amber-500/50" },
    contempt: { bg: "bg-purple-500", shadow: "shadow-purple-500/50" },
    content: { bg: "bg-emerald-500", shadow: "shadow-emerald-500/50" },
    disgusted: { bg: "bg-lime-500", shadow: "shadow-lime-500/50" },
    excited: { bg: "bg-pink-500", shadow: "shadow-pink-500/50" },
    fear: { bg: "bg-cyan-500", shadow: "shadow-cyan-500/50" },
    sad: { bg: "bg-indigo-500", shadow: "shadow-indigo-500/50" },
    surprised: { bg: "bg-orange-500", shadow: "shadow-orange-500/50" },
    neutral: { bg: "bg-slate-500", shadow: "shadow-slate-500/50" },
  }

  return (
    <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 p-6 backdrop-blur-sm">
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <BarChart className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-semibold">Distribution</h3>
        </div>
        <p className="text-sm text-muted-foreground">Last {Math.min(20, analyses.length)} analyses</p>
      </div>

      <div className="space-y-3">
        {Object.entries(emotionCounts)
          .sort(([, a], [, b]) => b - a)
          .map(([emotion, count]: any) => {
            const colorData = emotionColors[emotion] || { bg: "bg-primary", shadow: "shadow-primary/50" }
            return (
              <div key={emotion} className="group">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-medium capitalize">{emotion}</span>
                  <span className="text-xs font-bold text-primary">{count}</span>
                </div>
                <div className="h-2.5 bg-muted/50 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 group-hover:shadow-lg group-hover:${colorData.shadow} ${colorData.bg}`}
                    style={{
                      width: `${totalAnalyses > 0 ? (count / totalAnalyses) * 100 : 0}%`,
                    }}
                  />
                </div>
              </div>
            )
          })}
      </div>
    </Card>
  )
}

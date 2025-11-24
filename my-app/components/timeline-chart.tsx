"use client"

import { Card } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

export function TimelineChart({ analyses }: { analyses: any[] }) {
  const chartData = analyses
    .slice()
    .reverse()
    .slice(0, 10)
    .map((a, idx) => ({
      id: idx,
      timestamp: new Date(a.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      [a.primary_emotion]: Math.round(a.confidence * 100),
    }))

  return (
    <Card className="bg-gradient-to-br from-card to-card/50 border-border/50 p-6 backdrop-blur-sm">
      <h3 className="text-lg font-semibold mb-4">Emotion Timeline</h3>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="timestamp" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="confidence" stroke="#8b5cf6" dot={{ fill: "#8b5cf6" }} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

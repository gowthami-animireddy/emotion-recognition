"use client"

import { useState } from "react"
import { AnalysisPanel } from "./analysis-panel"
import { HistoryChart } from "./history-chart"
import { DashboardHeader } from "./dashboard-header"
import { InputModalities } from "./input-modalities"
import { EmotionMetrics } from "./emotion-metrics"
import { Sparkles } from "lucide-react"
import { EmotionDistributionChart } from "./emotion-distribution-chart"
import { TimelineChart } from "./timeline-chart"

export function Dashboard() {
  const [analyses, setAnalyses] = useState<any[]>([])
  const [currentAnalysis, setCurrentAnalysis] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleAnalysis = async (data: any) => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
      const result = await response.json()

      const newAnalysis = {
        id: Date.now(),
        timestamp: new Date(),
        input_type: data.input_type,
        ...result.data,
      }

      setAnalyses([newAnalysis, ...analyses])
      setCurrentAnalysis(newAnalysis)
    } catch (error) {
      console.error("Analysis error:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-background via-background to-background">
      {/* Enhanced Sidebar */}
      <div className="w-80 border-r border-border/50 bg-gradient-to-b from-card to-background backdrop-blur-sm">
        <DashboardHeader />
        <InputModalities onAnalyze={handleAnalysis} isLoading={isLoading} />
      </div>

      {/* Main Content with Premium Layout */}
      <div className="flex-1 overflow-auto">
        <div className="p-8 lg:p-12">
          {currentAnalysis ? (
            <div className="space-y-6 max-w-7xl mx-auto">
              {/* Top Section: Main Analysis */}
              <AnalysisPanel analysis={currentAnalysis} />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <EmotionDistributionChart emotions={currentAnalysis.all_emotions} />
                {analyses.length > 0 && <TimelineChart analyses={analyses} />}
              </div>

              {/* Grid Layout: Metrics and History */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <EmotionMetrics analysis={currentAnalysis} />
                </div>
                <div>
                  <HistoryChart analyses={analyses} />
                </div>
              </div>

              {/* Recent Analyses Summary */}
              {analyses.length > 0 && (
                <div className="mt-8 pt-8 border-t border-border/30">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    Recent Analyses
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
                    {analyses.slice(0, 5).map((analysis) => (
                      <button
                        key={analysis.id}
                        onClick={() => setCurrentAnalysis(analysis)}
                        className="p-3 rounded-lg bg-card/50 hover:bg-card/80 border border-border/50 hover:border-primary/50 transition-all"
                      >
                        <p className="text-xs text-muted-foreground mb-1">
                          {new Date(analysis.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                        <p className="text-sm font-semibold capitalize text-primary">{analysis.primary_emotion}</p>
                        <p className="text-xs text-muted-foreground">{(analysis.confidence * 100).toFixed(0)}%</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="mb-4 p-4 rounded-full bg-primary/10">
                <Sparkles className="w-12 h-12 text-primary" />
              </div>
              <h2 className="text-2xl font-bold mb-2">Start Analyzing Emotions</h2>
              <p className="text-muted-foreground max-w-md">
                Upload text, audio, or video on the left to analyze emotions using our advanced CNN-RNN hybrid model
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

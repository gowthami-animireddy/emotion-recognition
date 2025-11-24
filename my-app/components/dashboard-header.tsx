import { Brain, Sparkles } from "lucide-react"

export function DashboardHeader() {
  return (
    <div className="p-6 border-b border-border">
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2 bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg border border-primary/30">
          <Brain className="w-6 h-6 text-transparent bg-gradient-to-r from-primary to-secondary bg-clip-text" />
        </div>
        <div className="flex-1">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent flex items-center gap-2">
            Emotion AI
            <Sparkles className="w-5 h-5 text-secondary" />
          </h1>
        </div>
      </div>
      <p className="text-sm text-muted-foreground ml-11">Multi-modal emotion recognition with CNN-RNN hybrid model</p>
    </div>
  )
}

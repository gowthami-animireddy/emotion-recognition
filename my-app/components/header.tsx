"use client"

export function Header() {
  return (
    <header className="border-b border-border bg-background/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-lg">ER</span>
          </div>
          <h1 className="text-2xl font-bold text-foreground">EmotionRecog</h1>
        </div>
        <nav className="flex items-center gap-6">
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition">
            Documentation
          </a>
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition">
            API
          </a>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition font-medium text-sm">
            Get Started
          </button>
        </nav>
      </div>
    </header>
  )
}

export function HeroSection() {
  return (
    <section className="relative overflow-hidden py-20">
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />
      <div className="container mx-auto px-4 text-center relative z-10">
        <h2 className="text-5xl font-bold text-foreground mb-6 text-balance">Understand Emotions Across Modalities</h2>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8 text-pretty">
          Recognize and analyze emotions from text, audio, and video using advanced hybrid CNN-RNN architectures
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          <div className="px-6 py-3 bg-primary/10 rounded-lg border border-primary/20 text-sm font-medium text-primary">
            Text Analysis
          </div>
          <div className="px-6 py-3 bg-secondary/10 rounded-lg border border-secondary/20 text-sm font-medium text-secondary">
            Audio Recognition
          </div>
          <div className="px-6 py-3 bg-accent/10 rounded-lg border border-accent/20 text-sm font-medium text-accent">
            Video Detection
          </div>
        </div>
      </div>
    </section>
  )
}

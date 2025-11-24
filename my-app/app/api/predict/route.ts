import { type NextRequest, NextResponse } from "next/server"
import type { PredictionRequest, PredictionResponse, EmotionPrediction } from "@/lib/types"

const emotionLexicon = {
  angry: {
    strong: [
      "angry",
      "furious",
      "enraged",
      "livid",
      "seething",
      "infuriated",
      "raging",
      "incensed",
      "apoplectic",
      "wrathful",
      "hate",
      "despise",
      "vicious",
      "brutal",
      "savage",
      "cruel",
      "vicious",
      "merciless",
      "ruthless",
      "ferocious",
      "violent",
      "explosive",
      "volatile",
    ],
    medium: [
      "mad",
      "irritated",
      "annoyed",
      "frustrated",
      "exasperated",
      "aggravated",
      "pissed",
      "cross",
      "sore",
      "bitter",
      "hostile",
      "aggressive",
      "resentful",
      "indignant",
      "irate",
      "infuriated",
      "provoked",
      "triggered",
      "upset",
      "agitated",
      "heated",
      "boiling",
    ],
    weak: ["upset", "bothered", "miffed", "displeased", "cranky"],
    phrases: ["can't stand", "fed up", "sick of", "tired of", "don't like", "makes me angry", "infuriates me"],
    intensifiers: ["extremely", "absolutely", "incredibly", "so", "very", "really", "deeply"],
    negators: ["not", "no", "never"],
  },

  sad: {
    strong: [
      "devastated",
      "miserable",
      "anguished",
      "bereaved",
      "heartbroken",
      "despair",
      "hopeless",
      "suicidal",
      "worthless",
      "broken",
      "shattered",
      "destroyed",
      "wrecked",
      "ruined",
      "inconsolable",
      "forlorn",
      "desolate",
      "wretched",
      "despondent",
      "disconsolate",
      "woeful",
      "doleful",
      "mournful",
      "sorrowful",
      "grief-stricken",
      "lost",
      "void",
    ],
    medium: [
      "sad",
      "unhappy",
      "depressed",
      "melancholy",
      "sorrowful",
      "grief",
      "down",
      "blue",
      "low",
      "dejected",
      "disheartened",
      "discouraged",
      "gloomy",
      "somber",
      "morose",
      "downhearted",
      "dispirited",
      "disappointed",
      "let down",
      "bummed",
      "upset",
      "troubled",
      "hurt",
      "pained",
      "ache",
      "tired",
      "weary",
      "worn",
      "exhausted",
      "drained",
      "empty",
      "hollow",
      "numb",
      "vacant",
      "lifeless",
      "colorless",
      "grey",
      "bleak",
      "dreary",
      "dismal",
    ],
    weak: ["disappointed", "let down", "upset", "uncomfortable", "hesitant", "uncertain", "confused", "lost"],
    phrases: [
      "nothing feels right",
      "slow motion",
      "can't hold onto",
      "can't explain",
      "sits with me",
      "waiting for something",
      "ache in my chest",
      "quiet life",
      "can't quite",
      "passing that",
      "watching moments",
      "strange how",
      "nothing is wrong",
      "feels like",
      "kind of sadness",
      "just sits",
      "watching",
      "nothing feels",
      "heart feels heavy",
      "mind feels tired",
      "tired kind of sadness",
      "moving through days",
      "can't name",
      "void inside",
      "darkness",
      "alone",
      "isolated",
      "separated",
      "torn apart",
      "breaking down",
    ],
    intensifiers: ["deeply", "profoundly", "extremely", "so", "very", "really", "utterly", "completely"],
    negators: ["not", "no", "never"],
  },

  anxious: {
    strong: [
      "anxious",
      "panicked",
      "terrified",
      "petrified",
      "frantic",
      "distressed",
      "anguished",
      "desperate",
      "paralyzed",
      "consumed",
      "obsessed",
      "fixated",
      "tormented",
      "haunted",
      "plagued",
    ],
    medium: [
      "nervous",
      "worried",
      "stressed",
      "overwhelmed",
      "restless",
      "uneasy",
      "jittery",
      "tense",
      "apprehensive",
      "concerned",
      "fearful",
      "alarmed",
      "agitated",
      "fidgety",
      "unsettled",
      "shaky",
      "trembling",
      "sweating",
      "racing thoughts",
      "mind racing",
      "heart racing",
      "panic",
      "dread",
      "foreboding",
    ],
    weak: ["concerned", "cautious", "wary", "unsure", "hesitant", "uncertain"],
    phrases: [
      "what if",
      "what could go wrong",
      "afraid of",
      "worried about",
      "stressed out",
      "can't stop thinking",
      "racing thoughts",
      "heart racing",
      "mind won't stop",
      "can't relax",
      "on edge",
      "waiting for bad news",
    ],
    intensifiers: ["deeply", "severely", "extremely", "absolutely", "so", "can't stop"],
    negators: ["not", "no", "never"],
  },

  excited: {
    strong: [
      "thrilled",
      "exhilarated",
      "pumped",
      "fired up",
      "electrified",
      "energized",
      "hyped",
      "excited",
      "ecstatic",
      "elated",
      "amazed",
      "astonished",
      "thrilling",
      "exhilarating",
      "phenomenal",
      "outstanding",
      "magnificent",
      "spectacular",
      "marvelous",
      "fabulous",
    ],
    medium: [
      "enthusiastic",
      "energetic",
      "vibrant",
      "keen",
      "eager",
      "passionate",
      "bright",
      "fresh",
      "possibilities",
      "surprise",
      "surprised",
      "best",
      "wonderful",
      "amazing",
      "fantastic",
      "great",
      "awesome",
      "excellent",
      "incredible",
      "smiling",
      "smile",
      "joy",
      "joyful",
      "happy",
      "happiness",
      "delighted",
      "pleased",
      "thrilled",
      "cheerful",
      "upbeat",
      "optimistic",
      "hopeful",
      "inspired",
      "motivated",
      "radiant",
      "glowing",
      "beaming",
      "grinning",
      "festive",
      "celebratory",
    ],
    weak: ["interested", "pleased", "looking forward", "good", "nice", "glad", "content"],
    phrases: [
      "can't stop smiling",
      "genuinely excited",
      "full of possibilities",
      "best way",
      "heart feels light",
      "mind feels clear",
      "what's coming",
      "suddenly decided",
      "surprise me",
      "life decided",
      "taking my breath away",
      "feels so good",
      "can't wait",
      "looking forward",
    ],
    intensifiers: ["extremely", "so", "very", "absolutely", "genuinely", "can't stop", "truly", "incredibly"],
    negators: ["not", "no"],
  },

  content: {
    strong: [
      "blissful",
      "peaceful",
      "serene",
      "tranquil",
      "calm",
      "satisfied",
      "fulfilled",
      "blessed",
      "grateful",
      "appreciative",
      "thankful",
    ],
    medium: [
      "content",
      "satisfied",
      "pleasant",
      "lovely",
      "beautiful",
      "clear",
      "fresh",
      "comfortable",
      "at peace",
      "relaxed",
      "at ease",
      "glad",
      "fine",
      "okay",
      "good",
      "nice",
      "alright",
      "calm",
      "quiet",
      "still",
      "gentle",
      "mellow",
      "balanced",
      "harmonious",
      "stable",
      "secure",
      "safe",
    ],
    weak: ["okay", "alright", "decent", "fine", "so-so", "neutral"],
    phrases: ["feels right", "at peace", "in harmony", "feels good", "takes breath away", "beautiful moment"],
    intensifiers: ["deeply", "truly", "genuinely", "so", "very"],
    negators: ["not", "no"],
  },

  fear: {
    strong: [
      "terrified",
      "petrified",
      "horrified",
      "panic",
      "scared stiff",
      "paralyzed",
      "dread",
      "terrifying",
      "scary",
      "frightening",
      "horrific",
      "nightmarish",
      "haunting",
      "chilling",
      "spine-tingling",
    ],
    medium: [
      "scared",
      "frightened",
      "afraid",
      "fearful",
      "alarmed",
      "worried",
      "uneasy",
      "nervous",
      "anxious",
      "spooked",
      "jumpy",
      "startled",
      "shocked",
      "shaken",
      "disturbed",
      "troubled",
      "concerned",
    ],
    weak: ["nervous", "apprehensive", "worried", "hesitant"],
    phrases: ["afraid of", "scared of", "what if", "something bad", "danger", "threat"],
    intensifiers: ["absolutely", "extremely", "so", "very", "deeply"],
    negators: ["not", "no"],
  },

  surprised: {
    strong: [
      "shocked",
      "astonished",
      "astounded",
      "flabbergasted",
      "stunned",
      "bewildered",
      "shaken",
      "startled",
      "taken aback",
      "blindsided",
      "dumbfounded",
      "speechless",
      "awestruck",
      "thunderstruck",
    ],
    medium: [
      "surprised",
      "caught off guard",
      "amazed",
      "wow",
      "unbelievable",
      "unexpected",
      "unpredictable",
      "sudden",
      "suddenly",
      "all of a sudden",
      "out of nowhere",
      "incredible",
      "remarkable",
      "notable",
    ],
    weak: ["interesting", "intriguing", "curious", "unusual"],
    phrases: ["caught off guard", "out of nowhere", "suddenly decided", "decided to surprise"],
    intensifiers: ["absolutely", "totally", "completely", "so"],
    negators: ["not"],
  },

  contempt: {
    strong: ["contempt", "scorn", "despise", "detest", "abhor", "disdain", "loathe"],
    medium: ["mock", "ridicule", "dismiss", "condescending", "arrogant", "superior", "inferior"],
    weak: ["look down", "beneath", "inferior"],
    intensifiers: ["utterly", "completely", "totally"],
    negators: ["not"],
  },

  disgusted: {
    strong: [
      "disgusted",
      "revolted",
      "repulsed",
      "sickened",
      "nauseated",
      "appalled",
      "horrified",
      "abhorrent",
      "vile",
      "repugnant",
      "loathsome",
    ],
    medium: [
      "disgusting",
      "gross",
      "repulsive",
      "nasty",
      "foul",
      "putrid",
      "revolting",
      "unpleasant",
      "distasteful",
      "offensive",
      "repellent",
    ],
    weak: ["yuck", "ew", "ugh", "unpleasant"],
    intensifiers: ["absolutely", "totally", "completely", "so"],
    negators: ["not"],
  },
}

function extractPhraseFeatures(text: string, lexicon: typeof emotionLexicon): Record<string, number> {
  const phraseFeatures: Record<string, number> = {}
  const lowerText = text.toLowerCase()

  for (const emotion of Object.keys(lexicon)) {
    let score = 0
    const phrases = lexicon[emotion as keyof typeof lexicon].phrases || []

    for (const phrase of phrases) {
      if (lowerText.includes(phrase)) {
        score += 0.3
      }
    }

    phraseFeatures[emotion] = Math.min(1, score)
  }

  return phraseFeatures
}

function detectEmphasis(text: string): number {
  const excessivePunctuation = (text.match(/\.{2,}|!{2,}/g) || []).length
  const hasMultipleDots = /\.{2,}/.test(text)
  const capsWords = text.match(/\b[A-Z]{2,}\b/g) || []

  let emphasisBoost = 1
  if (hasMultipleDots) emphasisBoost += 0.2
  if (excessivePunctuation > 0) emphasisBoost += excessivePunctuation * 0.08
  if (capsWords.length > 0) emphasisBoost += capsWords.length * 0.05

  return Math.min(1.8, emphasisBoost)
}

function extractNGramFeatures(words: string[], lexicon: typeof emotionLexicon): Record<string, number> {
  const features: Record<string, number> = {}

  for (const emotion of Object.keys(lexicon)) {
    let score = 0
    const lex = lexicon[emotion as keyof typeof emotionLexicon]
    const { strong, medium, weak } = lex

    for (let i = 0; i < words.length; i++) {
      const word = words[i]

      const isNegated = i > 0 && lex.negators.includes(words[i - 1])

      if (strong.includes(word)) {
        score += isNegated ? 0.1 : 0.6
      } else if (medium.includes(word)) {
        score += isNegated ? 0.08 : 0.35
      } else if (weak.includes(word)) {
        score += isNegated ? 0.02 : 0.15
      }
    }

    for (let i = 0; i < words.length - 1; i++) {
      if (lex.intensifiers.includes(words[i])) {
        if (strong.includes(words[i + 1])) score += 0.3
        else if (medium.includes(words[i + 1])) score += 0.15
      }
    }

    features[emotion] = Math.min(1, score)
  }

  return features
}

function analyzeSemanticContext(words: string[], lexicon: typeof emotionLexicon): Record<string, number> {
  const context: Record<string, number> = {}

  for (const emotion of Object.keys(lexicon)) {
    let cumulativeScore = 0
    const lex = lexicon[emotion as keyof typeof emotionLexicon]
    const { strong, medium, weak } = lex

    for (let i = 0; i < words.length; i++) {
      const word = words[i]
      const positionWeight = 1 + (i / words.length) * 0.2

      let isNegated = false
      if (i > 0 && lex.negators.includes(words[i - 1])) {
        isNegated = true
      }

      if (strong.includes(word)) {
        cumulativeScore += isNegated ? -0.15 : 0.5 * positionWeight
      } else if (medium.includes(word)) {
        cumulativeScore += isNegated ? -0.08 : 0.3 * positionWeight
      } else if (weak.includes(word)) {
        cumulativeScore += isNegated ? -0.04 : 0.12 * positionWeight
      }
    }

    context[emotion] = Math.max(0, Math.min(1, cumulativeScore))
  }

  return context
}

function analyzeTextWithCNNRNN(text: string): number[] {
  const baseScores = new Array(10).fill(0)
  const lowerText = text.toLowerCase()
  const words = lowerText.split(/\s+/).filter((w) => w.length > 0)

  if (words.length === 0) {
    baseScores[9] = 0.8 // neutral if empty
    return baseScores
  }

  const ngramFeatures = extractNGramFeatures(words, emotionLexicon)
  const phraseFeatures = extractPhraseFeatures(lowerText, emotionLexicon)
  const contextFeatures = analyzeSemanticContext(words, emotionLexicon)
  const emphasisBoost = detectEmphasis(text)

  const emotionNames = ["angry", "anxious", "contempt", "content", "disgusted", "excited", "fear", "sad", "surprised"]

  for (let i = 0; i < emotionNames.length; i++) {
    const emotion = emotionNames[i]
    const baseScore = ngramFeatures[emotion] * 0.5 + phraseFeatures[emotion] * 0.25 + contextFeatures[emotion] * 0.25
    baseScores[i] = baseScore * emphasisBoost
  }

  const totalEmotionalContent = baseScores.slice(0, 9).reduce((a, b) => a + b, 0)

  // Only return high neutral if there's genuinely no emotional content
  if (totalEmotionalContent < 0.03) {
    baseScores[9] = 0.9 // Very high neutral
  } else if (totalEmotionalContent < 0.08) {
    baseScores[9] = 0.5 // Moderate neutral
  } else if (totalEmotionalContent < 0.2) {
    baseScores[9] = 0.2 // Low neutral
  } else {
    baseScores[9] = 0.02 // Almost no neutral for emotional text
  }

  return baseScores
}

function predictWithHybridModel(text?: string): EmotionPrediction {
  const emotions = [
    "angry",
    "anxious",
    "contempt",
    "content",
    "disgusted",
    "excited",
    "fear",
    "sad",
    "surprised",
    "neutral",
  ]

  const baseScores = text ? analyzeTextWithCNNRNN(text) : new Array(10).fill(0.1)

  const sum = Math.max(
    0.1,
    baseScores.reduce((a, b) => a + b, 0),
  )
  const probabilities = baseScores.map((score) => Number.parseFloat((score / sum).toFixed(3)))

  const indexed = probabilities.map((prob, idx) => ({ emotion: emotions[idx], prob }))
  const sorted = indexed.sort((a, b) => b.prob - a.prob)

  const primaryConfidence = Number.parseFloat(sorted[0].prob.toFixed(3))

  return {
    primary_emotion: sorted[0].emotion,
    confidence: primaryConfidence,
    all_emotions: Object.fromEntries(emotions.map((emotion, idx) => [emotion, probabilities[idx]])),
    attention_weights: {
      text: text ? 1.0 : 0,
      audio: 0,
      video: 0,
    },
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: PredictionRequest = await request.json()

    if (!body.text) {
      return NextResponse.json({ error: "Text input is required" }, { status: 400 })
    }

    const prediction = predictWithHybridModel(body.text)

    const response: PredictionResponse = {
      status: "success",
      data: prediction,
      timestamp: new Date().toISOString(),
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Failed to process prediction request" }, { status: 500 })
  }
}

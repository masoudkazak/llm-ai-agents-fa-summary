# Prompt Engineering

Prompt engineering is the practice of designing and refining inputs to language models in order to get accurate, relevant, and useful outputs. A well-crafted prompt can dramatically improve model performance without changing the model itself.

---

## Elements of a Prompt

A prompt can contain one or more of these building blocks:

| Element | Description | Example |
|---------|-------------|---------|
| **Context** | Background information that helps the model understand the situation | "You are a senior software engineer reviewing a pull request." |
| **Input Data** | The actual content the model should process | A block of code, a document, a user message |
| **Instruction** | A clear description of what the model should do | "Summarize the following text." |
| **Examples** | Sample input/output pairs that demonstrate the expected behavior | "Input: cat → Output: animal" |
| **Constraints** | Rules that limit or shape the output | "Reply in under 100 words." / "Only use bullet points." |

Not every prompt needs all five elements — use whichever combination produces the best result for your use case.

---

## Prompting Techniques

### 1. Zero-Shot Learning
Ask the model to perform a task with no examples. Works well for simple or common tasks.

> "Translate this sentence to French: The weather is nice today."

### 2. Few-Shot Learning
Provide a few examples before the actual task. The model infers the pattern and applies it. Uses more tokens but significantly improves accuracy on complex or unusual tasks.

> "Positive: I loved this movie! → Sentiment: positive  
> Negative: The food was terrible. → Sentiment: negative  
> Now classify: The hotel was okay. → Sentiment: ?"

### 3. Instruction Prompting
Give the model explicit, structured instructions rather than relying on examples. Especially useful when you need a specific format or reasoning style.

> "Read the following customer review and do three things:
> 1. Identify the main complaint.
> 2. Rate the sentiment from 1 to 5.
> 3. Suggest one improvement the company could make."

---

## Why Prompts Fail

Even well-intentioned prompts can produce poor results. Common causes:

1. **Vague prompts** — the model doesn't have enough direction to know what a "good" answer looks like. Fix: be specific about the task, format, and audience.
2. **Bias and leading language** — phrasing that pushes the model toward a predetermined answer, reducing objectivity. Fix: use neutral language.
3. **Evolving language and knowledge** — the model's training data has a cutoff date, so it may be unaware of recent events, terminology, or domain changes. Fix: provide up-to-date context in the prompt.

---

## Mitigation Strategies

When a prompt consistently fails, you have three main options:

1. **Improve the prompt** — the first and cheapest option. Clarify instructions, add examples, or restructure the prompt.
2. **Fine-tune the foundation model** — train the model on domain-specific data so it learns the patterns you care about. Higher cost and complexity, but can yield large gains.
3. **Solve the problem a different way** — sometimes the right move is to redesign the pipeline. Break the task into smaller steps, use retrieval-augmented generation (RAG) to inject fresh context, or handle edge cases with deterministic code instead of relying on the model.

---

## Chain of Thought (CoT) Prompting

Rather than asking the model to jump straight to an answer, you instruct it to reason through a series of steps first. This is especially effective for multi-step or logical tasks.

**Example:**

> Walk me through the process of planning a weekly vegan menu.
>
> - Step 1: Describe healthy ingredient selection
> - Step 2: Describe a balanced meal using those ingredients
> - Step 3: Describe the specific time of day to eat the meal
> - Step 4: Describe exercises that complement the diet for optimal health

**Why it works:** Language models generate text token by token. When you ask for intermediate reasoning steps, the model "thinks out loud" and is less likely to skip to a wrong conclusion.

---

## Augmenting Prompts

LLMs learn from training data, but they have a knowledge cutoff and no access to your private or real-time data. Augmenting a prompt means injecting relevant, up-to-date information directly into the prompt before sending it.

**Common augmentation sources:**
- Internal documents or knowledge bases
- Database query results
- API responses (weather, prices, inventory, etc.)
- Structured data like JSON or CSV

This is the core idea behind **Retrieval-Augmented Generation (RAG)**: retrieve relevant data at runtime and include it in the prompt. It's often a better choice than fine-tuning when your data changes frequently.

---

## Prompt Tuning vs. Prompt Engineering

These terms are often confused:

| | **Prompt Engineering** | **Prompt Tuning** |
|---|---|---|
| **What it is** | Designing a prompt from scratch for a task | Iteratively refining an existing prompt to improve output |
| **Who does it** | Any developer or user | Usually done systematically, sometimes automated |
| **Model changes** | No | No |
| **When to use** | Starting a new task | Improving an existing, partially-working prompt |

Both are different from **fine-tuning**, which actually updates the model's weights.

---

## Effective Prompts for Image Generation

Text-to-image models (like DALL·E, Midjourney, or Stable Diffusion) respond best when prompts include:

1. **Subject** — who or what is in the image
2. **Description** — scene, lighting, mood, colors, spatial relationships
3. **Style** — the visual aesthetic (e.g., photorealistic, oil painting, flat illustration, anime)

**Example:**
> A lone astronaut sitting on a cliff edge overlooking a vast alien desert at sunset, photorealistic, cinematic lighting, 8K

---

## Controlling Output Length (Reducing Generated Tokens)

Generating fewer tokens reduces cost and latency. Three main approaches:

1. **Stop Sequences** — a specific string or token that tells the model to stop generating when it appears.
   > In `"recommend three books in the mystery genre"`, the word `three` implicitly limits scope — but you can also pass an explicit stop string like `"\n\n"` to halt after the first paragraph.

2. **`max_tokens` parameter** — hard cap on the number of tokens the model can produce. Any response is cut off at this limit.

3. **`n` and `stream` parameters**
   - **`n`**: the number of independent completions to generate for a single prompt. Useful when you want multiple candidate answers to compare. Note: cost scales with `n`.
   - **`stream`**: instead of waiting for the full response, tokens are sent to the client as they are generated. This doesn't reduce total tokens but improves perceived latency in user-facing applications.

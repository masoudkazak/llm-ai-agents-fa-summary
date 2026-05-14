# Practical LLM & AI Agent Notes

This repository contains a cleaned, structured, and expanded version of my personal learning notes.

## Audience

Python developers building LLM apps (OpenAI API, Hugging Face, LangChain/LangGraph, Gradio demos, basic fine-tuning, and agent frameworks).

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

If you want one large environment (not recommended for production, fine for learning):

```bash
pip install \
  openai gradio pillow requests httpx python-dotenv pydantic ipython \
  transformers accelerate bitsandbytes torch datasets pypdf \
  langchain langchain-openai langchain-community chromadb tiktoken \
  langgraph graphviz tavily-python amadeus \
  crewai pandas numpy matplotlib seaborn scikit-learn xgboost \
  pyautogen google-generativeai anthropic \
  openai-agents "gradio[mcp]"
```

Environment variables you will typically need:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `GOOGLE_API_KEY` (Gemini)
- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`
- `AMADEUS_API_KEY`, `AMADEUS_API_SECRET`

## Topics

- [Audience](#audience)
- [Quick Setup](#quick-setup)
- [09. OpenAI API Basics](#09-openai-api-basics)
- [12. Understanding the Response Object and Tokens](#12-understanding-the-response-object-and-tokens)
- [15. Giving the Model a Personality (System Prompt)](#15-giving-the-model-a-personality-system-prompt)
- [20. Image Inputs Prerequisite](#20-image-inputs-prerequisite)
- [23. Prompt Components and Techniques](#23-prompt-components-and-techniques)
- [26. Vision Example (Base64 Image + Text)](#26-vision-example-base64-image--text)
- [35. Gradio + OpenAI (Demo UI)](#35-gradio--openai-demo-ui)
- [39. Gradio Interface Anatomy](#39-gradio-interface-anatomy)
- [41. Minimal Gradio App](#41-minimal-gradio-app)
- [42. Streaming Responses](#42-streaming-responses)
- [45. Explanation-Level Slider (Gradio)](#45-explanation-level-slider-gradio)
- [50. Vellum](#50-vellum)
- [53. Chatbot Arena](#53-chatbot-arena)
- [56. Using Gemini/Anthropic](#56-using-geminianthropic)
- [74. Why Hugging Face](#74-why-hugging-face)
- [77. Hugging Face Core Libraries and GPU Check](#77-hugging-face-core-libraries-and-gpu-check)
- [80. Transformers: Pipeline, Tokenizer, Model](#80-transformers-pipeline-tokenizer-model)
- [83. Tokenization Example](#83-tokenization-example)
- [86. Quantization vs dtype vs device_map + CausalLM](#86-quantization-vs-dtype-vs-device_map--causallm)
- [89. Reading PDFs with PyPDF](#89-reading-pdfs-with-pypdf)
- [92. PDF Assistant (Text Generation QA)](#92-pdf-assistant-text-generation-qa)
- [95. Multi-Model Gradio + Freeing GPU Memory](#95-multi-model-gradio--freeing-gpu-memory)
- [103. Working with Datasets (Hugging Face)](#103-working-with-datasets-hugging-face)
- [106. Two Training Paradigms (High-Level)](#106-two-training-paradigms-high-level)
- [110. Steps to Choose the Right LLM](#110-steps-to-choose-the-right-llm)
- [120. RAG Definition](#120-rag-definition)
- [Day 07 - Build a RAG Pipeline in LangChain](#day-07---build-a-rag-pipeline-in-langchain)
- [Day 08 - Resume Assistant (OpenAI/Gemini + Pydantic)](#day-08---resume-assistant-openaigemini--pydantic)
- [Day 09 - Fine-Tuning Open-Source Models (PEFT/LoRA)](#day-09---fine-tuning-open-source-models-peftlora)
- [Day 10 - Multi-Model Agent Teams with AutoGen](#day-10---multi-model-agent-teams-with-autogen)
- [Day 11 - Agentic Workflows in LangGraph](#day-11---agentic-workflows-in-langgraph)
- [Day 12 - CrewAI + Predictive Analytics (Regression)](#day-12---crewai--predictive-analytics-regression)
- [Day 13 - MCP + OpenAI Agents SDK](#day-13---mcp--openai-agents-sdk)

---

## 09. OpenAI API Basics

Correct API shape uses `messages` (plural) and it must be a list.

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Explain recursion in one paragraph."}
    ],
)

ai_reply_content = response.choices[0].message.content
print(ai_reply_content)
```

Roles:

- `user`: the user message
- `assistant`: model response
- `system`: instructions that set behavior

## 12. Understanding the Response Object and Tokens

Typical fields you will see:

- `id`: request/response id
- `model`: model name
- `choices[0].finish_reason`: why generation stopped (`stop`, `length`, ...)
- `created`: timestamp
- `usage.prompt_tokens`: tokens in your input
- `usage.completion_tokens`: tokens in the model output
- `usage.total_tokens`: prompt + completion

Token intuition:

- Tokens are units of text (not words).
- Rough English heuristic: ~0.75 words/token (varies).

## 15. Giving the Model a "Personality" (System Prompt)

If you want consistent behavior, put it in `system`.

```python
messages = [
    {
        "role": "system",
        "content": "You are a concise tutor. Explain clearly and ask one clarifying question if needed.",
    },
    {"role": "user", "content": "Teach me Big-O notation."},
]
```

## 20. Image Inputs Prerequisite

For image processing helpers in Python, install Pillow:

```bash
pip install pillow
```

## 23. Prompt Components and Techniques

Prompt components (practical template):

1. Context: who the assistant is
2. Instruction: the task + constraints
3. Input data: the text/data you provide
4. Output indicator: required structure

Techniques:

- Zero-shot: ask directly, no examples
- Few-shot: include a few examples
- Decomposition (often called chain-of-thought style planning): split complex tasks into smaller steps

Example educational prompt:

```
System: You are a careful tutor. Explain step-by-step, but keep the final answer concise.
User: Explain {topic}. Include: (1) definition, (2) one analogy, (3) one short code example, (4) common mistake.
Output: Markdown with headings: Definition, Analogy, Example, Common Mistake.
```

## 26. Vision Example (Base64 Image + Text)

Reusable helper + query function (corrected from shorthand):

```python
import base64
import io
import os
from typing import Union

from PIL import Image
from openai import OpenAI


def encode_image_to_base64(image: Union[str, Image.Image]) -> str:
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image_format = (image.format or "JPEG").upper()
        image.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    raise TypeError("image must be a file path or PIL.Image")


def query_openai_vision(client: OpenAI, image: Union[str, Image.Image], prompt: str,
                       model: str = "gpt-4o", max_tokens: int = 200) -> str:
    b64 = encode_image_to_base64(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
```

Vision prompt (educational):

```
You are a visual reasoning assistant.
Task: describe the scene, list key objects, then answer the question.
Constraints: if you are unsure, say what is unclear.
Question: {question}
```

## 35. Gradio + OpenAI (Demo UI)

Gradio is a fast way to demo ML/LLM apps with a web interface.

Temperature note:

- `temperature` is usually between `0` and `2`.
- `0.7` is a common balance between deterministic and creative.

### Why low temperature for document QA?

Document QA should be factual and grounded in the provided text; low temperature reduces creative drift.

Corrected call shape:

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

system_prompt = "You are a helpful tutor."
message = "Explain what a vector database is."

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ],
    temperature=0.7,
)
```

## 39. Gradio Interface Anatomy

`gr.Interface` parameters:

- `fn`: Python function (pass the function object, no parentheses)
- `inputs`: input components (Textbox, Slider, ...)
- `outputs`: output components
- `title`, `description`: UI text

## 41. Minimal Gradio App

```python
import gradio as gr


def tutor(question: str) -> str:
    return f"You asked: {question}"


demo = gr.Interface(
    fn=tutor,
    inputs=gr.Textbox(lines=4, placeholder="Ask a question", label="Question"),
    outputs=gr.Textbox(label="Answer"),
    title="AI Tutor",
    description="A minimal Gradio demo",
)

demo.launch()
```

## 42. Streaming Responses

When you stream, return incremental output with `yield`.

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")


def stream_answer(system_prompt: str, message: str):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            full_response += delta
            yield full_response
```

Streaming with Gradio (generator function):

```python
import gradio as gr


EXPLANATION_LEVELS = {
    1: "Like I'm 5 years old.",
    2: "Like I'm 10 years old.",
    3: "Like I'm a high school student.",
    4: "Like I'm a college student.",
    5: "Like I'm an expert in the field.",
}


def make_system_prompt(level: int) -> str:
    level_desc = EXPLANATION_LEVELS.get(level, "Explain clearly.")
    return f"You are a friendly technical tutor. {level_desc} Use examples and avoid unnecessary jargon."


def tutor_stream(question: str, level: int):
    system_prompt = make_system_prompt(level)
    for partial in stream_answer(system_prompt, question):
        yield partial


demo = gr.Interface(
    fn=tutor_stream,
    inputs=[
        gr.Textbox(lines=4, label="Question"),
        gr.Slider(1, 5, step=1, value=3, label="Explanation Level"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="AI Tutor (Streaming)",
)

demo.launch()
```

## 45. Explanation-Level Slider (Gradio)

Define levels and inject them into the prompt (best in the system message). This section is the non-streaming form of the same idea.

## 50. Vellum

Vellum is a platform to compare/evaluate LLMs and prompts.

## 53. Chatbot Arena

Chatbot Arena is a community benchmark where you can compare model outputs side-by-side and vote.

## 56. Using Gemini/Anthropic

After obtaining API keys:

```bash
pip install google-generativeai anthropic
```

## 74. Why Hugging Face

Reasons (from the notes):

1. More control and customization
2. Lower cost using smaller models
3. Open-source architectures are visible
4. Strong community
5. Offline / on-prem possibilities

## 77. Hugging Face Core Libraries and GPU Check

Install:

- `transformers`
- `accelerate`
- `bitsandbytes`
- `torch`
- `pypdf`

GPU check example:

```python
import torch

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    torch.set_default_device("cuda")
    print("PyTorch default device set to CUDA")
else:
    print("Warning: No GPU detected. Running on CPU will be extremely slow!")
```

How to login in code:

```python
import os
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))
```

Why we use `bitsandbytes`:

- Enables 4-bit/8-bit quantization to reduce VRAM
- Makes larger models feasible on limited GPUs

## 80. Transformers: Pipeline, Tokenizer, Model

Three core concepts:

1. `pipeline(...)`: easy high-level wrapper
2. `AutoTokenizer`: automatically downloads the correct tokenizer
3. `AutoModelFor...`: automatically downloads the model architecture + weights

Pipeline example:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="gpt2")
print(pipe("Hello", max_new_tokens=20)[0]["generated_text"])
```

## 83. Tokenization Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
encoded = tokenizer("Hello world")
print(encoded["input_ids"])  # list of token IDs (not words)
```

## 86. Quantization vs dtype vs device_map + CausalLM

Key ideas:

- `load_in_4bit` / `load_in_8bit`: how weights are stored (memory)
- `torch_dtype`: compute precision during operations
- `device_map="auto"`: let Accelerate place layers on available devices
- `return_tensors`: output tensor type (`pt`, `tf`, `np`)

`skip_special_tokens`:

- When decoding, removes special tokens like `<eos>`, `<pad>`, etc.

‍`return_tensors`:
- `pt`: PyTorch tensors
- `tf`: TensorFlow tensors
- `np`: NumPy arrays

BitsAndBytesConfig common knobs:

- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.float16` (or bfloat16)

About `trust_remote_code=True`:

- Only enable for trusted model repos that require custom code.

Two generation styles:

1. Manual tokenize -> `generate` -> decode
2. `pipeline("text-generation", ...)`

Example (manual):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    quantization_config=quantization_config,
)

prompt = "Write a short poem about rain."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

Example (pipeline):

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
out = pipe("Hello", max_new_tokens=20, temperature=0.7)
print(out[0]["generated_text"])
```

## 89. Reading PDFs with PyPDF

```python
import pypdf

reader = pypdf.PdfReader("file.pdf")
num_pages = len(reader.pages)
page_text = reader.pages[0].extract_text()
```

## 92. PDF Assistant (Text Generation QA)

The note uses a text-generation pipeline. The key is:

- extract text
- truncate to a limit
- build a prompt template
- generate
- parse out the answer

Expanded and corrected version:

```python
MAX_CONTENT_CHARS = 25_000


def answer_question_from_pdf(document_text: str, question: str, llm_pipeline):
    context = (document_text[:MAX_CONTENT_CHARS] + "...") if len(document_text) > MAX_CONTENT_CHARS else document_text

    prompt_template = f"""<|system|>
You are a document QA assistant.
Use ONLY the provided document text.
If the answer is not in the document, say: I don't know.

Document text:
{context}
<|end|>
<|user|>
Question:
{question}
<|end|>
<|assistant|>
Answer:"""

    outputs = llm_pipeline(
        prompt_template,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.1,
        top_p=1.0,
    )

    full_text = outputs[0]["generated_text"]
    answer_start = full_text.find("Answer:")
    raw = full_text[answer_start + len("Answer:") :].strip() if answer_start != -1 else full_text.strip()

    end_token = "<|end|>"
    if end_token in raw:
        raw = raw.split(end_token)[0].strip()

    return raw
```

What are `do_sample` and `top_p`?

- `do_sample=False`: deterministic generation (less random)
- `do_sample=True`: sampling for diversity
- `top_p`: nucleus sampling; lower values reduce randomness (only relevant when sampling)

## 95. Multi-Model Gradio + Freeing GPU Memory

The note describes a pattern: in a Gradio dropdown you switch models; when you switch, release the old model/tokenizer and free GPU cache.

Key ideas:

- Keep a single "current" model loaded
- On change: delete old references, `gc.collect()`, and `torch.cuda.empty_cache()`

```python
import gc
import torch

current_model = None
current_tokenizer = None


def unload_current():
    global current_model, current_tokenizer
    current_model = None
    current_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Dropdown switching sketch:

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CHOICES = ["gpt2", "distilgpt2"]


def load_model(model_id: str):
    unload_current()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return model, tokenizer


def on_model_change(model_id: str):
    global current_model, current_tokenizer
    current_model, current_tokenizer = load_model(model_id)
    return f"Loaded {model_id}"


with gr.Blocks() as demo:
    dropdown = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
    status = gr.Markdown()
    dropdown.change(on_model_change, inputs=[dropdown], outputs=[status])

demo.launch()
```

## 103. Working with Datasets (Hugging Face)

```python
from datasets import load_dataset

dataset = load_dataset("dataset_id", split="train")
print(dataset.features)
print(dataset.to_pandas().head())
print(dataset.select(range(10)).to_pandas())
```

## 106. Two Training Paradigms (High-Level)

- SFT (Supervised Fine-Tuning): learn from prompt -> correct answer pairs
- RL-style training: learn by optimizing a reward signal

## 110. Steps to Choose the Right LLM

1. Define the core task (summarization, classification, search, QA, ...)
2. Define expected input/output
3. Identify constraints: privacy, latency, cost, time-to-market
4. Choose model type and size (you may not need a generative model)
5. Choose open vs closed source
6. Decide scaling/deployment strategy

## 120. RAG Definition

RAG (Retrieval-Augmented Generation):

- Retrieve relevant external information
- Add retrieved chunks to the prompt
- Generate an answer grounded in that context

---

## Day 07 - Build a RAG Pipeline in LangChain

LangChain lets you combine components (loaders, splitters, embeddings, vector stores, retrievers, chains) to build RAG systems.

Install:

```bash
pip install langchain langchain-openai langchain-community chromadb tiktoken
```

Concepts:

- `chunk_size`: chunk length (often characters)
- `chunk_overlap`: how much overlap between chunks

Embeddings definition:

- Embeddings represent text as dense numeric vectors so similarity search can find related chunks.

Example pipeline (module imports vary by LangChain version):

```python
from langchain_openai import OpenAIEmbeddings

# Depending on your LangChain version, these may live in langchain_community.
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

DATA_FILE_PATH = "data.txt"

loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
documents = splitter.split_documents(raw_docs)

embeddings = OpenAIEmbeddings(api_key="YOUR_API_KEY")
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

# Optional: inspect one stored embedding/document (internal API; learning/debug only)
stored = vector_store._collection.get(include=["embeddings", "documents"], limit=1)
print(len(stored["embeddings"][0]))
print(stored["documents"][0][:200])

# Retrieval test
query = "What is the refund policy?"
similar_docs = vector_store.similarity_search(query, k=2)
print(similar_docs[0].page_content)
```

Build QA chain (corrected name is `RetrievalQAWithSourcesChain`):

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0, api_key="YOUR_API_KEY")

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",  # put retrieved chunks into the prompt
    retriever=retriever,
    return_source_documents=False,
    verbose=False,
)

result = qa_chain.invoke({"question": query})
print(result.get("answer"))
print(result.get("sources"))
```

Notes:

- `return_source_documents=True` includes the source docs in the result object.
- `verbose=True` is useful for debugging.

---

## Day 08 - Resume Assistant (OpenAI/Gemini + Pydantic)

Install:

```bash
pip install pydantic openai google-generativeai python-dotenv ipython
```

Pattern:

1. Create a gap analysis between job description and resume
2. Generate an updated resume and a diff
3. Enforce schema with Pydantic

## Why Pydantic here?

When an LLM returns "JSON", it is easy to get:

- missing fields
- wrong types (e.g., list instead of string)
- extra keys you did not ask for
- invalid JSON (trailing commas, comments, etc.)

Pydantic solves the \"LLM output is untrusted\" problem by validating and normalizing the output against a Python schema.

Practical benefits:

- You get a strongly-typed object (`ResumeOutput`) instead of a loose dict.
- Validation errors are explicit and debuggable (you can retry the LLM with the error message).
- Downstream code becomes safer (no `KeyError` / `None` surprises).

## Core concepts (quick)

- `BaseModel`: define a schema with type hints.
- Validation: Pydantic checks types and required fields at runtime.
- Serialization: convert a model back to JSON/dict for storage or APIs.

Example schema:

```python
from pydantic import BaseModel, Field


class ResumeOutput(BaseModel):
    updated_resume: str = Field(..., description="Full updated resume text")
    diff_markdown: str = Field(..., description="Markdown diff vs original resume")
```

### Parsing model output safely

If your LLM returns a JSON string, parse and validate it.

Pydantic v2:

```python
obj = ResumeOutput.model_validate_json(json_string_from_llm)
data = obj.model_dump()
```

Pydantic v1:

```python
obj = ResumeOutput.parse_raw(json_string_from_llm)
data = obj.dict()
```

If validation fails, catch the exception and retry with a stricter prompt:

```python
from pydantic import ValidationError

try:
    obj = ResumeOutput.model_validate_json(json_string_from_llm)  # v2
except ValidationError as e:
    # Send e.errors() back to the model and ask it to fix the JSON.
    raise
```

Example (structure-focused, with explicit prompts):

```python
from pydantic import BaseModel


def analyze_resume_against_job_description(job_description_text: str, resume_text: str) -> str:
    prompt = f"""You are a career coach.
Compare the resume against the job description.
Return a bullet gap analysis with: missing keywords, missing experience signals, and suggested rewrites.

Job description:\n{job_description_text}\n\nResume:\n{resume_text}\n"""
    gap_analysis = openai_generate(prompt, temperature=0.7)
    return gap_analysis


class ResumeOutput(BaseModel):
    updated_resume: str
    diff_markdown: str


def generate_resume(job_description_text: str, resume_text: str, gap_analysis_openai: str) -> dict:
    prompt = f"""You are a resume editor.
Use the gap analysis to improve the resume.
Return JSON with fields: updated_resume, diff_markdown.

Gap analysis:\n{gap_analysis_openai}\n\nOriginal resume:\n{resume_text}\n"""
    updated_resume_json = openai_generate(prompt, temperature=0.7, response_format=ResumeOutput)
    return updated_resume_json
```

---

## Day 09 - Fine-Tuning Open-Source Models (PEFT/LoRA)

Motivation example (from the notes):

- A generic model answers medical questions vaguely.
- Fine-tuning on medical data can make answers more specific and useful.

Libraries:

```bash
pip install transformers accelerate bitsandbytes torch datasets peft trl scikit-learn gradio
```

Concepts:

- PEFT: adapt models efficiently without updating all weights
- TRL: training utilities for transformer models (including SFT)
- GPU is practically required for meaningful fine-tuning speed
- `seed`: makes randomness reproducible
- `shuffle`: randomize ordering during splitting/training

Load and split dataset:

```python
import torch
from datasets import load_dataset

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    torch.set_default_device("cuda")
else:
    print("Warning: No GPU detected. Running on CPU will be extremely slow!")

labeled_dataset = load_dataset("dataset_id", split="train")

split_dataset = labeled_dataset.train_test_split(
    test_size=0.1,
    seed=42,
    shuffle=True,
)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
```

Format examples for SFT (corrected + explicit prompt):

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

system_prompt = (
    "You are a sentiment classifier. "
    "Return ONLY one label: Positive, Neutral, or Negative."
)


def format_for_sft_gemma(example, tokenizer: AutoTokenizer):
    user_prompt = f"Sentence: {example['text']}"
    assistant_response = example["sentiment"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": formatted_text}


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

Tokenizer padding note (common in causal LMs):

```python
tokenizer = AutoTokenizer.from_pretrained("your_model_id")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Formatting datasets for trainers (pattern from notes):

```python
sft_train_dataset = train_dataset.map(
    format_for_sft_gemma,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=list(train_dataset.features),
)
sft_test_dataset = test_dataset.map(
    format_for_sft_gemma,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=list(test_dataset.features),
)
```

Confusion matrix (conceptual reminder from notes):

- TP: predicted positive, actually positive
- TN: predicted negative, actually negative
- FP: predicted positive, actually negative
- FN: predicted negative, actually positive

Metrics correction (the original note mixed accuracy and F1):

- Accuracy: `(TP + TN) / (TP + TN + FP + FN)`
- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1: `2 * (Precision * Recall) / (Precision + Recall)`

Zero-shot classification with base model (pattern from notes, cleaned):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

labels = ["Positive", "Neutral", "Negative"]


def create_zeroshot_prompt(sentence: str) -> str:
    return (
        "You are a sentiment classifier. Return only one label: Positive, Neutral, or Negative.\n"
        f"Sentence: {sentence}\n"
        "Label:"
    )


def classify_zero_shot(sentence: str, model, tokenizer) -> str:
    prompt = create_zeroshot_prompt(sentence)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            do_sample=False,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    predicted = "Unknown"
    for label in labels:
        if label.lower() in response_text.lower():
            predicted = label
            break

    return predicted
```

Generation params reminders:

- `truncation=True`: if text is too long, cut it
- `max_length`: max tokens the model can see
- EOS: if model hits EOS token, stop generation

LoRA overview:

- reduces compute/memory by training small adapter matrices

### LoRA (Practical Tutorial Add-on)

LoRA (Low-Rank Adaptation) fine-tunes a model by injecting small trainable matrices into selected weight layers (usually attention projections), while keeping the original model weights frozen.

Why this is useful:

- Much lower GPU memory usage than full fine-tuning
- Faster training
- Easier to store/share adapters (only a small file)

Core intuition:

- A dense weight update `DeltaW` can be approximated as `A @ B` where `A` and `B` are low-rank matrices (`rank = r`).
- You train only `A` and `B` (the adapter), not the full `W`.

Key LoRA hyperparameters:

- `r`: rank (capacity). Typical: 4, 8, 16.
- `lora_alpha`: scaling factor. Common: 16, 32, 64.
- `lora_dropout`: regularization to reduce overfitting. Common: 0.0-0.1.
- `target_modules`: which layers to adapt (often `q_proj`, `v_proj` for LLaMA-like models).
- `bias`: whether to train bias terms (usually `\"none\"`).

When to use LoRA:

- You need task adaptation but cannot afford full fine-tuning.
- You want to maintain the base model and swap task adapters.
- You want to combine LoRA with quantization (QLoRA) for even smaller VRAM usage.

### QLoRA (LoRA + 4-bit quantization)

QLoRA loads the base model in 4-bit and trains LoRA adapters on top. In practice:\n
- Base weights are quantized (memory saving)\n
- Adapters are trained in higher precision\n
- You get good tradeoffs on limited GPUs\n

### Minimal PEFT LoRA Skeleton (Educational)

This is a reference pattern; exact imports and supported args can vary by model and library versions.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "your_model_id"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # adjust per architecture
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

Data formatting tip (matches your earlier SFT formatting style):

- Always keep a consistent chat template (`system`, `user`, `assistant`) for training.
- For classification tasks, force a constrained output (e.g., only one label).

Saving, loading, merging adapters (conceptual):

- Save adapters: `model.save_pretrained(\"adapter_dir\")`
- Load adapters later onto the same base model
- Merge for inference (if supported): adapter weights can be merged into base weights to simplify deployment

Training tooling terms:

- `SFTTrainer`: manages supervised fine-tuning on prompt->response data
- `TrainingArguments`: config for how training runs

Why repeating the same data improves performance (epochs):

- each epoch adjusts weights slightly toward correct outputs
- changes are small, so multiple passes help convergence
- even with same data, updated weights improve next pass

---

## Day 10 - Multi-Model Agent Teams with AutoGen

AutoGen orchestrates multiple agents that collaborate.

Install:

```bash
pip install -q pyautogen openai google-generativeai "ag2[gemini]"
```

Key terms:

- `chat manager`: an agent that coordinates other agents
- `human_input_mode`: whether an agent can ask for human input
- `code_execution_config`: whether code execution is allowed

`max_consecutive_auto_reply`:

- limits automatic replies in a row (helps prevent runaway loops)

Example (OpenAI + Gemini mix, cleaned and corrected):

```python
import autogen

openai_api_key = "YOUR_OPENAI_KEY"
google_api_key = "YOUR_GOOGLE_KEY"

config_list_openai = [{"model": "gpt-4o-mini", "api_key": openai_api_key}]
llm_config_openai = {"config_list": config_list_openai, "temperature": 0.7, "timeout": 120}

config_list_gemini = [{"model": "gemini-2.0-flash", "api_key": google_api_key, "api_type": "google"}]
llm_config_gemini = {"config_list": config_list_gemini, "temperature": 0.6, "timeout": 120}

cmo_prompt = "You are a CMO. Propose positioning and messaging."
brand_marketer_prompt = "You are a brand marketer. Create slogans and campaign ideas."

cmo_agent_gemini = autogen.ConversableAgent(
    name="Chief_Marketing_Officer_Gemini",
    system_message=cmo_prompt,
    llm_config=llm_config_gemini,
    human_input_mode="NEVER",
)

brand_marketer_agent_openai = autogen.ConversableAgent(
    name="Brand_Marketer_OpenAI",
    system_message=brand_marketer_prompt,
    llm_config=llm_config_openai,
    human_input_mode="NEVER",
)

user_proxy_agent = autogen.UserProxyAgent(
    name="Human_User_Proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1,
    is_termination_msg=lambda x: x.get("content", "").rstrip().lower() in {"exit", "quit"},
    code_execution_config=False,
)

for a in (cmo_agent_gemini, brand_marketer_agent_openai, user_proxy_agent):
    a.reset()

from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[user_proxy_agent, cmo_agent_gemini, brand_marketer_agent_openai],
    messages=[],
    max_round=20,
)

manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_openai)
manager.initiate_chat(recipient=brand_marketer_agent_openai, message="Launch a new coffee brand for students.")
```

---

## Day 11 - Agentic Workflows in LangGraph

LangGraph models agent workflows as nodes + edges with shared state.

Install:

```bash
pip install langchain langgraph langchain-openai tavily-python amadeus langchain-community graphviz
```

Key ideas (from notes, expanded):

- AI agents can act autonomously over time using tools
- `state` is a container passed between nodes
- nodes are steps (LLM call, tool call, transform)
- conditional edges route execution based on model output

Example 1: simple 2-node workflow (summarize -> translate)

```python
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    input_text: str
    summary: str
    translated_summary: str


def summarize_step(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = [
        ("system", "Summarize the text in exactly three sentences."),
        ("user", state["input_text"]),
    ]
    result = llm.invoke(prompt)
    return {**state, "summary": result.content}


def translate_step(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = [
        ("system", "Translate the summary into Persian. Keep technical terms consistent."),
        ("user", state["summary"]),
    ]
    result = llm.invoke(prompt)
    return {**state, "translated_summary": result.content}


workflow = StateGraph(AgentState)
workflow.add_node("summarize", summarize_step)
workflow.add_node("translate", translate_step)
workflow.add_edge("summarize", "translate")
workflow.set_entry_point("summarize")

app = workflow.compile()

initial_state: AgentState = {"input_text": "Lorem ipsum...", "summary": "", "translated_summary": ""}
final_state = app.invoke(initial_state)
print(final_state["translated_summary"])
```

Example 2: one tool + conditional edges (agent decides to use tool or stop)

```python
import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode


class ToolState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
search_tool = TavilySearchResults(max_results=3)


def call_model_with_tools(state: ToolState) -> ToolState:
    model_with_tools = llm.bind_tools([search_tool])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: ToolState) -> Literal["action", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "action"
    return END


graph = StateGraph(ToolState)
graph.add_node("agent", call_model_with_tools)
graph.add_node("action", ToolNode([search_tool]))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
graph.add_edge("action", "agent")

app = graph.compile()
final = app.invoke({"messages": [HumanMessage(content="What is 2+2?")]})
print(final["messages"][-1].content)
```

Notes:

- The model decides whether to use a tool.
- For simple tasks (like `2+2`) it may not call search.

Create custom tools:

- define a function and decorate it with `@tool` from `langchain.tools`

Flight search using Amadeus & ToolNode (cleaned and corrected from notes):

```python
from amadeus import Client, ResponseError
from langchain.tools import tool

amadeus_client = Client(
    client_id="AMADEUS_API_KEY",
    client_secret="AMADEUS_API_SECRET",
    hostname="test",
)


@tool
def search_flights_tool(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    travel_class: str = "ECONOMY",
    currency: str = "USD",
    max_offers: int = 5,
) -> str:
    """Search flights via Amadeus and return a compact text summary."""

    flight_search_params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": adults,
        "travelClass": travel_class,
        "currencyCode": currency,
        "max": max_offers,
    }
    if return_date:
        flight_search_params["returnDate"] = return_date

    try:
        response = amadeus_client.shopping.flight_offers_search.get(**flight_search_params)
        if not response.data:
            return "No flights found."

        results = []
        for offer in response.data[:max_offers]:
            price = offer.get("price", {}).get("total")
            itineraries = offer.get("itineraries", [])
            seg0 = (itineraries[0].get("segments") or [{}])[0] if itineraries else {}
            carrier = seg0.get("carrierCode")
            results.append(f"{carrier or '??'} - total {price or '??'} {currency}")

        return "\n".join(results)

    except ResponseError as e:
        return f"Amadeus error: {e}"
```

---

## Day 12 - CrewAI + Predictive Analytics (Regression)

The notes first review classic ML regression, then apply agent orchestration.

Key ML imports:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Pandas basics:

- `df.head()`, `df.tail()`, `df.info()`, `df.describe()`
- `df.describe(include=["object"])` for non-numerical summary
- `df.isnull().sum()`
- `df[col].unique()` and `df[col].nunique()`

Median imputation:

```python
median_imputer = SimpleImputer(strategy="median")
df[cols_with_missing] = median_imputer.fit_transform(df[cols_with_missing])
```

EDA steps (from notes, with cleaned code):

1) Target variable analysis

```python
plt.figure(figsize=(10, 5))
sns.histplot(df["Units Sold"], kde=True, bins=30)
plt.title("Units Sold Distribution")
plt.xlabel("Units Sold")
plt.ylabel("Frequency")
plt.show()

print(df["Units Sold"].skew())
```

2) Numerical feature analysis

```python
for col in numerical_features:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True, bins=25)
    plt.title(col)
    plt.show()

    plt.figure(figsize=(10, 2))
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```

3) Categorical feature analysis

```python
for col in categorical_features:
    plt.figure(figsize=(12, 5))
    sns.countplot(y=df[col], order=df[col].value_counts().index, palette="viridis")
    plt.title(col)
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.show()
```

4) Relationship analysis (numerical vs target, categorical vs target)

```python
for col in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[col], y=df["Units Sold"], alpha=0.5)
    plt.title(f"{col} vs Units Sold")
    plt.show()

for col in categorical_features:
    plt.figure(figsize=(14, 6))
    order = df.groupby(col)["Units Sold"].median().sort_values(ascending=False).index
    sns.boxplot(y=col, x="Units Sold", data=df, order=order, palette="Spectral")
    plt.title(f"{col} vs Units Sold")
    plt.show()
```

One-hot encoding note (from notes):

- Do not assign arbitrary integers to nominal categories; one-hot avoids fake ordering.

Preprocessing + split:

```python
df_processed = df.copy()
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

y = df_processed["Units Sold"]
X = df_processed.drop(columns=columns_to_drop)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
```

Models (from notes, cleaned):

```python
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred_lin = lin.predict(X_test)

mae_lin = mean_absolute_error(y_test, y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

import xgboost as xgb
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
```

Feature importance analysis (corrected attribute name):

```python
importances = rf.feature_importances_
feature_names = X_train.columns
sorted_importances = (
    pd.Series(importances, index=feature_names)
    .sort_values(ascending=False)
)
print(sorted_importances.head(20))
```

CrewAI install:

```bash
pip install crewai openai pandas langchain_openai
```

CrewAI imports:

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
```

CrewAI example (structure from notes; APIs vary by version):

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", api_key="YOUR_OPENAI_KEY")

planner_agent = Agent(role="Planner", goal="Plan the regression project", backstory="Senior DS", llm=llm, allow_delegation=False, verbose=True)
analyst_agent = Agent(role="Analyst", goal="Perform EDA and preprocessing", backstory="Data analyst", llm=llm, allow_delegation=False, verbose=True)
modeler_agent = Agent(role="Modeler", goal="Train and evaluate models", backstory="ML engineer", llm=llm, allow_delegation=False, verbose=True)

planning_task = Task(description="Plan approach and metrics", expected_output="A step-by-step plan", agent=planner_agent)
data_analysis_task = Task(description="Clean data and run EDA", expected_output="EDA summary and cleaned dataset", agent=analyst_agent)
modeling_task = Task(description="Train models and evaluate", expected_output="Metrics and recommendations", agent=modeler_agent)

regression_crew = Crew(
    agents=[planner_agent, analyst_agent, modeler_agent],
    tasks=[planning_task, data_analysis_task, modeling_task],
    process=Process.sequential,
    verbose=1,
)

crew_result = regression_crew.kickoff()
print(crew_result)
```

---

## Day 13 - MCP + OpenAI Agents SDK

MCP (Model Context Protocol) is an open standard for connecting models to tools/APIs/data in a structured and secure way.

Install (from notes):

```bash
pip install openai-agents gradio openai requests httpx pillow "gradio[mcp]"
```

Important fixes vs the raw notes:

- use `yield` (not `yeild`)
- Gradio keyword is `label` (not `lable`)
- in the summarizer, pass `text` (not `question`)

MCP server (cleaned and corrected from notes):

```python
from typing import Generator

import gradio as gr
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
client = OpenAI(api_key="YOUR_OPENAI_KEY")

EXPLANATION_LEVELS = {
    1: "Like I'm 5 years old",
    2: "Like I'm 10 years old",
    3: "Like a high school student",
    4: "Like a college student",
    5: "Like an expert in the field",
}


def explain_concept(question: str, level: int) -> Generator[str, None, None]:
    if not question.strip():
        yield "Error: question cannot be blank."
        return

    level_desc = EXPLANATION_LEVELS.get(level, "Clearly and concisely")
    system_prompt = f"You are a helpful tutor. Explain {level_desc}."

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
        temperature=0.7,
    )

    partial = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial


def summarize_text(text: str, compression_ratio: float = 0.3) -> Generator[str, None, None]:
    if not text.strip():
        yield "Error: text cannot be blank."
        return

    ratio = max(0.1, min(float(compression_ratio), 0.8))
    system_prompt = f"Summarize the user's text to about {int(ratio * 100)}% of its length. Keep key details."

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=True,
        temperature=0.5,
    )

    partial = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            partial += delta
            yield partial


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# AI Tutor (MCP)")

        with gr.Tab("Explain Concept"):
            q = gr.Textbox(label="Concept / Question")
            lvl = gr.Slider(1, 5, value=3, step=1, label="Explanation Level")
            out1 = gr.Markdown()
            gr.Button("Explain").click(explain_concept, inputs=[q, lvl], outputs=out1)

        with gr.Tab("Summarize Text"):
            txt = gr.Textbox(lines=8, label="Long Text")
            ratio = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Compression Ratio")
            out2 = gr.Markdown()
            gr.Button("Summarize").click(summarize_text, inputs=[txt, ratio], outputs=out2)

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", mcp_server=True)
```

MCP client schema fetch (cleaned from notes):

```python
import json
import httpx

MCP_BASE = "http://localhost:7860/gradio_api/mcp/sse"


def fetch_schema(server_url: str) -> dict:
    schema_url = server_url.replace("/sse", "/schema")
    with httpx.Client() as client:
        resp = client.get(schema_url, timeout=10)
        resp.raise_for_status()
        return resp.json()


schema = fetch_schema(MCP_BASE)
print(json.dumps(schema, indent=2))
```

Create an agent using OpenAI Agents SDK with MCP tools (outline from notes):

```python
import asyncio

from agents import Agent, Runner
from agents.mcp import MCPServerSse

MCP_BASE = "http://localhost:7860/gradio_api/mcp/sse"

mcp_tool = MCPServerSse({
    "name": "AI Tutor",
    "url": MCP_BASE,
    "timeout": 30,
    "client_session_timeout_seconds": 60,
})


async def main():
    agent = Agent(
        name="TutorAgent",
        instructions="Use the MCP tools when needed. Be concise.",
        model="gpt-4o-mini",
        mcp_servers=[mcp_tool],
    )

    await mcp_tool.connect()

    result = None
    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        if result is not None:
            new_input = result.to_input_list() + [{"role": "user", "content": user_input}]
        else:
            new_input = [{"role": "user", "content": user_input}]

        result = await Runner.run(starting_agent=agent, input=new_input)
        print(result.final_output)

        for msg in result.to_input_list():
            if "arguments" in msg:
                print(msg.get("name"), msg.get("arguments"))


if __name__ == "__main__":
    asyncio.run(main())
```

---

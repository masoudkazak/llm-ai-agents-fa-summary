# LLM & AI Agent Engineering

---

## Table of Contents

1. **OpenAI API**
   - Basic Chat Completions call, response structure, tokens
   - Roles (user, assistant, system) and giving the model a personality
2. **Prompting, Vision & Gradio**
   - Sending images to OpenAI (Vision), prompt components, techniques (zero-shot, few-shot, chain-of-thought)
   - Gradio UI, streaming, explanation-level slider
   - Model comparison tools (vellum.ai, Chatbot Arena), Google/Anthropic
3. **Hugging Face & Open-Source Models**
   - Why Hugging Face, required libraries (transformers, accelerate, bitsandbytes, torch, pypdf)
   - Login, GPU detection, bitsandbytes and quantization
   - Pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, return_tensors
   - PDF loading (pypdf), PDF Q&A assistant, do_sample and top_p
4. **Datasets, Training Types & Model Selection**
   - Loading and inspecting datasets on Hugging Face
   - SFT vs RL, steps to choose the right LLM for an application
5. **RAG & LangChain**
   - RAG (Retrieve, Augment, Generate), LangChain overview
   - Document loader, RecursiveCharacterTextSplitter, chunk_size and chunk_overlap
   - Chroma vector store, embeddings, similarity search, RetrievalQAWithSourcesChain
6. **Resume Assistant with Pydantic**
   - Pydantic for structured output and validation
   - Gap analysis between resume and job description, generating updated resume with structured response
7. **Fine-Tuning with PEFT & LoRA**
   - Intuition, PEFT, TRL, data formatting for SFT, confusion matrix and metrics
   - Zero-shot evaluation, EOS and generation params
   - **LoRA explained:** idea, low-rank matrices, rank and alpha, when to use, integration with SFTTrainer
   - SFTTrainer, TrainingArguments, why repeated data improves the model
8. **AutoGen: Multi-Agent Teams**
   - ConversableAgent, UserProxyAgent, human_input_mode, code_execution_config
   - max_consecutive_auto_reply, is_termination_msg
   - GroupChat, GroupChatManager, initiate_chat
9. **LangGraph: Agentic Workflows**
   - AI agents, state and StateGraph, nodes and edges
   - Example: summarize then translate; agent with tools and conditional edges
   - Custom tools (@tool), Amadeus flight search example
10. **CrewAI & Machine Learning**
    - Predictive analytics, regression (pandas, sklearn), EDA, one-hot encoding
    - Train/test split, LinearRegression, RandomForest, XGBoost, metrics, feature importance
    - CrewAI Agent, Task, Crew, Process.sequential, prompts importance
11. **MCP & OpenAI Agents SDK**
    - Model Context Protocol, MCP server with Gradio (streaming, launch with mcp_server=True)
    - MCP client, fetch schema, OpenAI Agents SDK Agent and Runner with MCP tools

---

## ۱. API (OpenAI)

### فراخوانی پایهٔ Chat Completions

```python
from openai import OpenAI

openai_client = OpenAI(api_key=openai_api_key)
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": message}]
)

ai_reply_content = response.choices[0].message.content
```

**نقش‌ها (roles):** `user`، `assistant`، `system`. نقش در پاسخ همان‌طور که شما در درخواست مشخص کرده‌اید برنمی‌گردد؛ مقدار واقعی را سرویس تنظیم می‌کند.

### ساختار پاسخ (Response)

| فیلد | توضیح |
|------|--------|
| `id` | شناسهٔ یکتای درخواست |
| `model` | نام مدل استفاده‌شده |
| `finish_reason` | دلیل پایان تولید (مثلاً `stop` یعنی به‌صورت عادی تمام شده) |
| `created` | timestamp ایجاد |
| `prompt_tokens` | تعداد توکن‌های ارسالی شما |
| `completion_tokens` | تعداد توکن‌های تولیدشده توسط مدل |
| `total_tokens` | جمع هر دو |

هر **token** در مدل به یک عدد نگاشت می‌شود. می‌توانید در صفحهٔ [Tokenizer](https://platform.openai.com/tokenizer) اپن‌ای‌آی تعداد توکن و نگاشت را ببینید. تقریباً **۱۰۰ توکن ≈ ۷۵ کلمه** در انگلیسی (یعنی حدوداً ۳/۴ کلمه به ازای هر توکن).

### دادن شخصیت به مدل (System Message)

برای تعریف نقش و شخصیت مدل، از پیام با نقش `system` استفاده کنید:

```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "تو یک معلم صبور هستی که مفاهیم را ساده توضیح می‌دهی."},
        {"role": "user", "content": mymessage}
    ]
)
```

---

## ۲. پرامپت، ویژن و Gradio

### ارسال تصویر به OpenAI (Vision)

- برای کار با تصویر در پایتون، پکیج **Pillow** را نصب کنید: `pip install pillow`.

### اجزای یک پرامپت خوب

1. **Context:** زمینه و شخصیت مدل نسبت به سؤال (چه کسی است و چه اطلاعاتی دارد).
2. **Instruction:** هدف دقیق و قواعدی که مدل باید رعایت کند.
3. **Input data:** خود دادهٔ ورودی (مثلاً متن یا تصویر).
4. **Output indicator:** فرمت یا ساختار خروجی مورد انتظار.

### تکنیک‌های پرامپت‌نویسی

| تکنیک | توضیح |
|--------|--------|
| **Zero-shot** | درخواست بدون مثال؛ مدل فقط با دستور جواب می‌دهد. |
| **Few-shot** | دادن چند مثال سؤال–جواب در خود پرامپت. |
| **Chain-of-Thought** | تقسیم مسئله یا وظیفهٔ پیچیده به مراحل کوچک‌تر تا مدل گام‌به‌گام فکر کند. |

### مثال: ارسال تصویر و متن به OpenAI Vision

```python
import io
import base64
import os
from PIL import Image

def encode_image_to_base64(image_path_or_pil):
    if isinstance(image_path_or_pil, str):
        if not os.path.exists(image_path_or_pil):
            raise FileNotFoundError(f"File not found: {image_path_or_pil}")
        with open(image_path_or_pil, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_path_or_pil, Image.Image):
        buffer = io.BytesIO()
        image_format = image_path_or_pil.format or "JPEG"
        image_path_or_pil.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError("Expected file path (str) or PIL Image")

def query_openai_vision(client, image, prompt, model="gpt-4o", max_tokens=100):
    """
    prompt مثال برای توصیف تصویر:
    "Describe this image in detail. List the main objects, colors, and any text visible."
    یا برای سؤال خاص: "What is written on the sign in this image?"
    """
    base64_image = encode_image_to_base64(image)
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling API: {e}"
```

### Gradio: دموی وب برای مدل

**Gradio** ابزاری است برای ساخت سریع رابط وب برای مدل‌های ML. نصب: `pip install gradio`.

- **`gr.Interface`** کلاس اصلی برای ساخت UI است.
- پارامترهای مهم: `fn` (تابع پایتون، بدون پرانتز)، `inputs`، `outputs`، `title`، `description`.

```python
import gradio as gr

SYSTEM_PROMPT_TUTOR = """You are a patient and clear tutor. Your role is to explain concepts so that the learner understands them.
Rules:
- Answer in the same language the user writes in (e.g. Persian or English).
- If the user asks a question, give a direct and structured answer.
- Use simple examples when helpful. Do not assume prior knowledge unless the user indicates otherwise.
- Be concise but complete. If the topic is complex, break it into short steps."""

def my_fn(message):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TUTOR},
            {"role": "user", "content": message},
        ],
        temperature=0.7,  # بین 0 و 2؛ حدود 0.7 برای تعادل خلاقیت و ثبات مناسب است
    )
    return response.choices[0].message.content

demo = gr.Interface(
    fn=my_fn,
    inputs=gr.Textbox(lines=5, placeholder="سؤال خود را بنویسید...", label="ورودی"),
    outputs=gr.Textbox(label="پاسخ"),
    title="دستیار هوشمند",
    description="یک سؤال بپرسید.",
)
demo.launch()
```

**Temperature:** میزان تصادفی بودن خروجی را کنترل می‌کند (۰ تا ۲). مقدار پایین‌تر پاسخ‌های قطعی‌تر و تکراری‌تر؛ مقدار بالاتر خلاق‌تر ولی ناپایدارتر. برای کارهای واقع‌گرایانه (مثل QA) معمولاً پایین (مثلاً ۰ یا ۰.۳) و برای ایده‌پردازی بالاتر (مثلاً ۰.۷) استفاده می‌شود.

### استریمینگ (Streaming)

با `stream=True` پاسخ به‌صورت تکه‌تکه (chunk) برمی‌گردد. در تابعی که به Gradio می‌دهید به‌جای `return` از **`yield`** استفاده کنید:

```python
def stream_response(message):
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TUTOR},
            {"role": "user", "content": message},
        ],
        temperature=0.7,
        stream=True,
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta and chunk.choices[0].delta.content:
            text_chunk = chunk.choices[0].delta.content
            full_response += text_chunk
            yield full_response
```

### اسلایدر برای سطح توضیح (Explanation Level)

می‌توانید با `gr.Slider` سطح سادگی توضیح را انتخاب کنید و آن را در **system prompt** (نه user prompt) استفاده کنید:

```python
explanation_levels = {
    1: "Like I'm 5 years old",
    2: "Like I'm 10 years old",
    3: "Like a high school student",
    4: "Like a college student",
    5: "Like an expert in the field",
}

def explain(message, level):
    level_desc = explanation_levels.get(level, "Clearly and concisely")
    system = f"""You are a tutor. Explain concepts at exactly this level: {level_desc}.
- Use the same language as the user (e.g. Persian or English).
- Do not use jargon above this level; if you must mention a term, define it briefly.
- Give one clear, coherent explanation. Use a short example if it helps."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content

demo = gr.Interface(
    fn=explain,
    inputs=[
        gr.Textbox(lines=5, placeholder="مفهومی که می‌خواهید توضیح داده شود", label="سؤال"),
        gr.Slider(minimum=1, maximum=5, step=1, value=3, label="سطح توضیح"),
    ],
    outputs=gr.Textbox(label="پاسخ"),
    title="معلم سطح‌پذیر",
    description="سؤال خود را بپرسید و سطح توضیح (از ساده تا تخصصی) را انتخاب کنید.",
)
demo.launch()
```

### ابزارهای مقایسهٔ مدل‌ها

- **vellum.ai:** مقایسهٔ مدل‌های زبانی.
- **Chatbot Arena:** امکان دادن پرامپت خودتان، مقایسهٔ خروجی مدل‌ها و رتبه‌دهی؛ رتبه‌بندی جمع‌سپاری کاربران هم وجود دارد.

### استفاده از Google و Anthropic

بعد از گرفتن API key مربوطه:

```bash
pip install google-generativeai anthropic
```

---

## ۳. Hugging Face و مدل‌های اوپن‌سورس

### چرا Hugging Face؟

1. **کنترل بیشتر** روی مدل و pipeline.
2. **مدل‌های کوچک‌تر** → هزینه و منابع کمتر.
3. **Open-source** و امکان مطالعهٔ معماری.
4. **جامعه (community)** و مدل/دیتاست آماده.
5. **اجرای offline** بدون وابستگی به API خارجی.

### کتابخانه‌های موردنیاز

| پکیج | کاربرد |
|------|--------|
| **transformers** | بارگذاری و استفادهٔ استاندارد از مدل‌ها و tokenizer از Hub |
| **accelerate** | اجرای کارآمد روی سخت‌افزارهای مختلف (GPU/CPU) |
| **bitsandbytes** | بارگذاری و اجرای مدل با حافظهٔ کمتر؛ مثلاً **۴-bit** یا **۸-bit** quantization |
| **torch** | فریم‌ورک deep learning (PyTorch) |
| **pypdf** | استخراج متن از PDF |

برای مدل‌های سنگین، **Google Colab** با GPU گزینهٔ مناسبی است.

### ورود به Hugging Face در کد (Login)

**سؤال:** چطور در کد به Hugging Face لاگین کنیم؟

با توکن دسترسی (Access Token) از [Settings → Access Tokens](https://huggingface.co/settings/tokens) در سایت HF:

```python
from huggingface_hub import login

login(token="YOUR_HF_TOKEN")
# یا بدون آرگومان؛ از شما در محیط اجرا توکن را می‌پرسد:
# login()
```

می‌توانید توکن را در متغیر محیطی `HF_TOKEN` قرار دهید و در کد از `os.environ.get("HF_TOKEN")` استفاده کنید تا در کد ذخیره نشود.

### تشخیص GPU و تنظیم دستگاه پیش‌فرض

```python
import torch

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    torch.set_default_device("cuda")
    print("PyTorch default device set to CUDA (GPU)")
else:
    print("Warning: No GPU detected. Running on CPU will be very slow!")
```

### چرا از bitsandbytes استفاده می‌کنیم؟

**سؤال:** چرا bitsandbytes؟

- مدل‌های بزرگ (مثلاً ۷B پارامتر) با دقت معمول (float32/float16) حافظهٔ زیادی می‌گیرند و روی GPUهای معمولی یا Colab رایگان جا نمی‌شوند.
- **Quantization** یعنی کاهش دقت ذخیره‌سازی وزن‌ها (مثلاً ۴-bit یا ۸-bit) تا حافظه کم شود و مدل روی سخت‌افزار محدود قابل اجرا باشد.
- **bitsandbytes** پیاده‌سازی کارآمد این نوع بارگذاری را در اختیار می‌گذارد و با `transformers` یکپارچه است.

### سه جزء اصلی در transformers

1. **`pipeline()`:** سطح بالا برای کارهای متداول مثل text generation، summarization و غیره.
2. **`AutoTokenizer`:** بارگذاری خودکار tokenizer مناسب هر مدل.
3. **`AutoModelForCausalLM`:** بارگذاری خودکار معماری و وزن‌های مدل برای تولید متن (Causal LM).

مثال pipeline ساده:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="MODEL_ID")
result = pipe("Your prompt here")
```

مثال tokenizer و تبدیل متن به شناسهٔ توکن‌ها:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MODEL_ID")
tokens = tokenizer("Your prompt")
print(tokens["input_ids"])  # لیست اعداد (توکن‌ها)
```

### تفاوت load_in_4bit / load_in_8bit و torch_dtype

**سؤال:** تفاوت این گزینه‌ها چیست؟

| پارامتر | معنی |
|---------|------|
| **torch_dtype="auto"** | انتخاب خودکار بهترین نوع (مثلاً float16 یا float32) بر اساس سخت‌افزار برای کارایی بهتر. |
| **load_in_4bit=True** | بارگذاری مدل با دقت ۴-bit (quantization) برای صرفه‌جویی شدید حافظه؛ مناسب Colab و GPUهای محدود. |
| **load_in_8bit=True** | مشابه بالا با دقت ۸-bit؛ حافظه کمتر از float16 ولی بیشتر از ۴-bit، کیفیت معمولاً بهتر از ۴-bit. |
| **device_map="auto"** | با کمک **accelerate**، لایه‌های مدل را خودکار روی GPU/CPU توزیع می‌کند. |

### skip_special_tokens چیست؟

**سؤال:** `skip_special_tokens` در decode چه کاری می‌کند؟

وقتی `tokenizer.decode(ids, skip_special_tokens=True)` استفاده می‌کنید، توکن‌های **ویژه** (مثل `[PAD]`, `[EOS]`, `[BOS]`, `<|endoftext|>` و توکن‌های مشابه مدل) در متن خروجی نمایش داده نمی‌شوند. فقط متن قابل‌خواندن برای انسان برمی‌گردد. برای خروجی نهایی مدل معمولاً `True` مناسب است.

### گزینه‌های مهم BitsAndBytesConfig

**سؤال:** گزینه‌های داخل BitsAndBytesConfig چه هستند؟

مثال متداول:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # نوع کوانتیزاسیون ۴-bit (nf4 معمولاً برای LLM بهتر است)
    bnb_4bit_compute_dtype=torch.float16,  # دقت محاسبات هنگام inference
    bnb_4bit_use_double_quant=True,  # کوانتیزاسیون دو مرحله‌ای برای صرفه‌جویی بیشتر (اختیاری)
)
```

- **load_in_4bit / load_in_8bit:** سطح کوانتیزاسیون.
- **bnb_4bit_quant_type:** مثلاً `"nf4"` برای مدل‌های مبتنی بر Normal Float ۴-bit.
- **bnb_4bit_compute_dtype:** دقت tensorها هنگام محاسبه (مثلاً float16).
- **bnb_4bit_use_double_quant:** اگر True باشد، ثابت‌های کوانتیزاسیون خودشان هم کوانتیز می‌شوند تا حافظه کمتر شود.

### return_tensors

خروجی tokenizer را مشخص می‌کند که به چه فرمتی برگردد:

- **`pt`:** PyTorch tensor
- **`tf`:** TensorFlow
- **`np`:** NumPy

برای استفاده با `model.generate()` در PyTorch معمولاً `return_tensors="pt"` می‌گذارید.

### بارگذاری مدل با AutoModelForCausalLM و کوانتیزاسیون

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,  # برای مدل‌های جدیدتر که کد سفارشی دارند ضروری است
)
```

دو روش استفاده:

**روش ۱ (مستقیم با tokenizer و model):**

```python
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**روش ۲ (با pipeline):**

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
outputs = pipe(prompt, max_new_tokens=256, temperature=0.7, do_sample=True)
response = outputs[0]["generated_text"]
```

### کار با PDF (pypdf)

```python
from pypdf import PdfReader

reader = PdfReader(pdf_file_path)
num_pages = len(reader.pages)
for page in reader.pages:
    text = page.extract_text()  # متن همان صفحه
```

### دستیار سؤال‌وـجواب از روی PDF

ایده: محدود کردن طول متن سند (به‌خاطر محدودیت context مدل)، قرار دادن متن در قالب پرامپت مدل و استخراج جواب از خروجی.

- **MAX_CONTENT_CHARS:** حداکثر تعداد کاراکتر متن PDF که در پرامپت می‌گذارید.
- در پرامپت می‌توانید زبان و سبک پاسخ را هم مشخص کنید (مثلاً برای Phi-3 از قالب `<|system|>`, `<|user|>`, `<|assistant|>` و غیره استفاده می‌شود).
- برای پاسخ‌های واقع‌گرایانه از سند، معمولاً **temperature** را پایین و **do_sample** را متناسب با مدل تنظیم می‌کنید.

نمونهٔ قالب پرامپت برای مدل‌های مبتنی بر chat (مثلاً Phi-3):

```python
MAX_CONTENT_CHARS = 8000  # متن سند را محدود کنید تا در context جا شود

def answer_question_from_pdf(document_text: str, question: str, llm_pipeline) -> str:
    context = document_text[:MAX_CONTENT_CHARS] + "..." if len(document_text) > MAX_CONTENT_CHARS else document_text

    prompt_template = """<|system|>
You are a helpful assistant. Answer the user's question using ONLY the document text below. Do not use external knowledge. If the answer is not in the document, say so clearly. Keep answers concise and in the same language as the question.
<|end|>
<|user|>
Document:
{context}

Question: {question}
<|end|>
<|assistant|>
Answer:""".format(context=context, question=question)

    outputs = llm_pipeline(
        prompt_template,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    full_text = outputs[0]["generated_text"]
    answer_start = full_text.find("Answer:") + len("Answer:")
    raw_answer = full_text[answer_start:].strip()
    if "<|end|>" in raw_answer:
        raw_answer = raw_answer.split("<|end|>")[0]
    return raw_answer
```

**do_sample و top_p در pipeline:**

- **do_sample=True:** مدل به‌جای انتخاب همیشه محتمل‌ترین توکن، از sampling استفاده می‌کند؛ خروجی متنوع‌تر و خلاق‌تر می‌شود.
- **do_sample=False:** همیشه گزینهٔ با بیشترین احتمال انتخاب می‌شود؛ خروجی قطعی و قابل تکرار.
- **top_p (nucleus sampling):** فقط از توکن‌هایی نمونه‌گیری می‌شود که مجموع احتمالشان تا حد معینی (مثلاً ۰.۹) باشد. برای کنترل تنوع و انسجام مفید است.

**توجه:** در خلاصهٔ اصلی یک typo بود: `RetrievalQAWithSrourcesChain`؛ شکل صحیح **`RetrievalQAWithSourcesChain`** است. همین‌طور **`RecursiveCharacterTextSplitter`** (نه Recyrsuve)، و **`AutoModelForCausalLM`** (نه CauselLM).

### انتخاب چند مدل در Gradio برای QA روی PDF

در یک درس نمونه، در Gradio امکان انتخاب بین چند مدل برای QA روی PDF اضافه شده بود. با تغییر مدل از منو، مدل و tokenizer قبلی از حافظه آزاد می‌شوند و در صورت استفاده از **torch**، cache گرافیکی هم پاک می‌شود تا مدل جدید با روش‌های قبلی (مثلاً quantization و device_map) بارگذاری شود.

---

## ۴. دیتاست‌ها، انواع آموزش و انتخاب مدل

### کار با دیتاست در Hugging Face

```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("dataset_id", split="train")
dataset.features   # توضیح ستون‌ها
dataset.to_pandas()  # تبدیل به DataFrame
dataset.select(range(10)).to_pandas()  # نمایش ۱۰ ردیف اول
```

### دو نوع آموزش مدل‌های زبانی

| نوع | توضیح | مثال |
|-----|--------|------|
| **Supervised Fine-Tuning (SFT)** | مدل با جفت سؤال–جواب درست آموزش می‌بیند. | رویکرد مشابه OpenAI |
| **Reinforcement Learning (RL)** | با سیگنال پاداش (Reward) و آزمون و خطا تنظیم می‌شود. | مثل DeepSeek |

### مراحل انتخاب مدل مناسب برای اپلیکیشن

1. **تعریف نیاز:** وظیفهٔ اصلی (خلاصه‌سازی، طبقه‌بندی، جستجو و ...)، ورودی/خروجی مورد انتظار، حریم خصوصی داده، تأخیر، هزینه، زمان عرضه.
2. **نوع مدل:** شاید برای کار شما Generative AI لازم نباشد؛ مطلب مرتبط در [cohere.com](https://cohere.com) را ببینید.
3. **اندازهٔ مدل:** متناسب با منابع و کیفیت مورد نیاز.
4. **منبع مدل:** Open vs Closed source.
5. **مقیاس‌پذیری:** چطور با رشد کاربر و ترافیک مقیاس می‌گیرد.

---

## ۵. RAG و LangChain

### RAG چیست؟

**RAG (Retrieval-Augmented Generation):**

1. **R (Retrieve):** جستجو در منبع خارجی (مثلاً مستندات) و بازیابی قطعات مرتبط.
2. **A (Augment):** اضافه کردن این قطعات به پرامپت کاربر.
3. **G (Generate):** تولید پاسخ با LLM روی پرامپت غنی‌شده.

### LangChain

**LangChain** فریم‌ورکی است برای ساخت اپلیکیشن‌های مبتنی بر LLM با چسباندن اجزای مختلف (لودر، تقسیم‌کننده، ذخیرهٔ برداری، زنجیرهٔ QA و ...) به هم.

نصب (نمونه):

```bash
pip install -q langchain langchain-openai chromadb tiktoken langchain-community
```

- **chunk_size:** حداکثر اندازهٔ هر قطعه متن (مثلاً بر اساس تعداد کاراکتر).
- **chunk_overlap:** تعداد کاراکتر مشترک بین دو قطعهٔ پشت‌سرهم تا ارتباط متن حفظ شود.

### استفادهٔ پایه: لودر، تقسیم، ذخیرهٔ برداری، جستجو

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")
raw_document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
documents = text_splitter.split_documents(raw_document)
# documents[0].page_content متن نخستین chunk است؛ هر document متادیتای دیگری هم دارد.

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)

# تست بازیابی:
similar_docs = vector_store.similarity_search(test_query, k=2)
```

**Embedding:** نمایش متن (کلمه، جمله، سند) به‌صورت بردار عددی متراکم است تا بتوان شباهت معنایی را با فاصله یا شباهت برداری اندازه گرفت.

### زنجیرهٔ RAG با LangChain

به **Retriever** (مثلاً همان vector store) و یک **LLM** (مثلاً OpenAI) نیاز دارید:

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# پرامپت پیش‌فرض زنجیره: context بازیابی‌شده و سؤال کاربر را می‌گیرد.
# می‌توان با chain_type_kwargs={"prompt": ...} پرامپت سفارشی داد، مثلاً:
# "Use only the following context to answer the question. If the answer is not in the context, say so. Context: {context} Question: {question}"

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm,
    chain_type="stuff",  # همهٔ متن‌های بازیابی‌شده در یک context قرار می‌گیرند
    retriever=retriever,
    return_source_documents=False,
    verbose=False,
)

result = qa_chain.invoke({"question": test_query})
answer = result.get("answer")
sources = result.get("sources")
```

- **return_source_documents=True:** علاوه بر متن، شیء سندهای منبع هم برگردانده می‌شود.
- **verbose=True:** مراحل اجرای زنجیره برای دیباگ چاپ می‌شود.

---

## ۶. دستیار رزومه با Pydantic

در این بخش از **OpenAI** و **Gemini** برای تحلیل رزومه در برابر آگهی شغلی و تولید رزومهٔ به‌روز استفاده می‌شود. **Pydantic** برای تعریف خروجی ساختاریافته (structured output) و اعتبارسنجی استفاده می‌شود.

### Pydantic در چند خط

**Pydantic** کتابخانه‌ای است برای تعریف مدل داده با type hints و اعتبارسنجی خودکار. به‌جای دیکشنری نامشخص، خروجی API را به صورت کلاس با فیلدهای مشخص تعریف می‌کنید؛ هم خوانایی بهتر هم خطای کمتر.

نصب:

```bash
pip install pydantic openai google-generativeai python-dotenv
```

### تحلیل فاصله (Gap) بین رزومه و آگهی

```python
from pydantic import BaseModel

def analyze_resume_against_job_description(job_description_text: str, resume_text: str) -> str:
    prompt = f"""You are an expert career coach. Compare the following job description with the candidate's resume.

Job description:
{job_description_text}

Candidate's resume:
{resume_text}

Provide a clear gap analysis:
1. What skills or experiences does the job require that the resume does not clearly show?
2. What strengths in the resume match the job well?
3. Concrete, actionable suggestions to improve the resume for this role (e.g. which keywords to add, which experience to highlight).
Write in a professional tone. Use the same language as the job description (e.g. English or Persian)."""
    gap_analysis = openai_generate(prompt, temperature=0.7)
    return gap_analysis

gap_analysis_openai = analyze_resume_against_job_description(job_description_text, resume_text)
```

### خروجی ساختاریافته با Pydantic

با تعریف یک کلاس Pydantic می‌توانید از API بخواهید خروجی را دقیقاً در قالب آن برگرداند (مثلاً با `response_format` در OpenAI):

```python
class ResumeOutput(BaseModel):
    updated_resume: str
    diff_markdown: str

def generate_resume(
    job_description_text: str,
    resume_text: str,
    gap_analysis_openai: str,
) -> dict:
    prompt = f"""You are an expert resume writer. Using the job description, current resume, and gap analysis below, produce:
1. An updated resume (full text) that better aligns with the job, incorporating the suggested improvements.
2. A short diff or summary in Markdown format showing what was added or changed (e.g. bullet points under each section).

Job description:
{job_description_text}

Current resume:
{resume_text}

Gap analysis and suggestions:
{gap_analysis_openai}

Output only valid JSON with two keys: "updated_resume" (string) and "diff_markdown" (string). Do not add any text outside the JSON."""
    updated_resume_json = openai_generate(
        prompt,
        temperature=0.7,
        response_format=ResumeOutput,
    )
    return updated_resume_json
```

در این الگو، خروجی به‌صورت شیء `ResumeOutput` اعتبارسنجی می‌شود و می‌توانید به‌راحتی به `updated_resume` و `diff_markdown` دسترسی داشته باشید.

---

## ۷. Fine-Tuning با PEFT و LoRA

### مثال شهودی

فرض کنید در شرکتی دارویی از LLaMA برای پاسخ به سؤالات استفاده می‌کنید. سؤال تخصصی پزشکی بپرسید؛ احتمالاً پاسخ کلی می‌دهد. اگر همان مدل را با مجموعه‌دادهٔ حوزهٔ پزشکی **fine-tune** کنید، پاسخ‌ها تخصصی‌تر می‌شوند.

### PEFT (Parameter-Efficient Fine-Tuning)

کتابخانه‌ای برای تطبیق مدل‌های بزرگ با کاربردهای مختلف **بدون** آموزش تمام پارامترها (که بسیار پرهزینه است). فقط بخشی از پارامترها یا لایه‌های اضافه آموزش داده می‌شوند.

### TRL (Transformer Reinforcement Learning)

کتابخانه‌ای که ابزارهای آموزش مدل‌های ترنسفورمر را با روش‌هایی مثل **SFT** و **RL** فراهم می‌کند.

نصب (نمونه):

```bash
pip install -q transformers accelerate bitsandbytes torch datasets peft trl scikit-learn gradio
```

برای این نوع آموزش، **GPU** تقریباً ضروری است. **seed** برای تکرارپذیری و **shuffle** برای به‌هم‌ریختن ترتیب داده در هر epoch استفاده می‌شود.

### قالب‌دهی داده برای SFT

داده را به فرمت گفتگو (user / assistant) و با **chat template** مدل (مثلاً Gemma) تبدیل می‌کنید:

- **apply_chat_template:** پیام‌ها را به قالب توکنی همان مدل تبدیل می‌کند.
- **tokenize=False:** خروجی متن بماند (نه لیست id).
- **add_generation_prompt=False:** برای آموزش لازم است؛ برای inference معمولاً True می‌گذارید تا مدل بداند باید جواب تولید کند.

برای مدل‌هایی مثل Gemma، اگر **pad_token** تعریف نشده باشد، معمولاً آن را برابر **eos_token** قرار می‌دهند.

### Confusion Matrix و معیارها

در طبقه‌بندی (مثلاً sentiment):

- **TP (True Positive):** مدل و واقعیت هر دو مثبت.
- **TN (True Negative):** مدل و واقعیت هر دو منفی.
- **FP (False Positive):** مدل مثبت، واقعیت منفی.
- **FN (False Negative):** مدل منفی، واقعیت مثبت.

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** ترکیب precision و recall است.

### Zero-Shot با مدل پایه

قبل از fine-tune می‌توان با همان مدل پایه و یک پرامپت مناسب، طبقه‌بندی zero-shot انجام داد و با معیارهای بالا ارزیابی کرد تا بعد از fine-tune مقایسه شود.

- **truncation=True:** اگر متن طولانی‌تر از حد مدل بود، قطع شود.
- **max_length:** حداکثر تعداد توکن ورودی.
- **EOS (End-of-Sequence):** وقتی مدل این توکن را تولید کرد، تولید متوقف می‌شود.

### LoRA (Low-Rank Adaptation) — آموزش کامل

**مشکل:** در Fine-Tuning معمولی، تمام وزن‌های مدل (مثلاً ۷ میلیارد پارامتر) به‌روز می‌شوند. این کار به حافظهٔ GPU بسیار زیاد و زمان آموزش طولانی نیاز دارد.

**ایدهٔ LoRA:** در لایه‌های ترنسفورمر (معمولاً attention)، ماتریس وزن اصلی **W** را در حین آموزش ثابت نگه می‌داریم و به‌جای آن دو ماتریس کوچک **A** و **B** را آموزش می‌دهیم و به‌صورت زیر استفاده می‌کنیم:

\[
\text{خروجی} = (W + B \cdot A) \cdot x
\]

- **W** ابعاد مثلاً `d × k` دارد (خیلی بزرگ).
- **A** ابعاد `r × k` و **B** ابعاد `d × r` دارند؛ **r** (rank) عددی کوچک است (مثلاً ۸، ۱۶، ۳۲).
- تعداد پارامترهای قابل آموزش: `r×(k+d)` که در مقایسه با `d×k` بسیار کمتر است.

**چرا Low-Rank؟** فرض این است که تغییرات مفید مدل برای تطبیق با وظیفهٔ جدید در یک زیرفضای با بعد کم (low-rank) قرار می‌گیرند؛ پس نیازی به تغییر تمام عناصر W نیست.

**پارامترهای مهم LoRA در PEFT:**

| پارامتر | معنی |
|---------|------|
| **r (rank)** | بعد ماتریس‌های A و B. بزرگ‌تر → ظرفیت بیشتر، پارامتر و هزینه بیشتر. معمولاً ۸ تا ۶۴. |
| **lora_alpha** | ضریب مقیاس برای به‌روزرسانی LoRA. معمولاً alpha = 2×rank یا برابر rank. |
| **target_modules** | لیست نام لایه‌هایی که LoRA روی آن‌ها اعمال شود (مثلاً `["q_proj","v_proj"]` در attention). |
| **lora_dropout** | dropout روی خروجی LoRA برای کاهش overfitting. |

**چه زمانی از LoRA استفاده کنیم؟** وقتی دیتاست نسبتاً کوچک است، وظیفهٔ جدید به دانش کلی مدل نزدیک است، یا منابع GPU محدود است. برای تغییرات بسیار بزرگ در رفتار مدل گاهی full fine-tune یا روش‌های دیگر مناسب‌ترند.

**ادغام با SFTTrainer:** در TRL می‌توان مدل را با `PeftModel` (از PEFT) بارگذاری کرد و همان مدل را به SFTTrainer داد؛ در این حالت فقط پارامترهای LoRA آموزش می‌بینند.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # بسته به معماری مدل متفاوت است
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # فقط پارامترهای LoRA trainable هستند
# بعد همین model را به SFTTrainer می‌دهید
```

### SFTTrainer و TrainingArguments

- **SFTTrainer:** فرایند آموزش نظارت‌شده (prompt → response) را مدیریت می‌کند.
- **TrainingArguments:** پارامترهایی مثل تعداد epoch، نرخ یادگیری، batch size و ... را تنظیم می‌کند.

### چرا با تکرار همان داده مدل بهتر می‌شود؟

- هر **epoch** یعنی یک بار دیدن کل دیتاست.
- در هر بار، وزن‌ها کمی به‌سمت کاهش خطا به‌روز می‌شوند؛ تغییر هر قدم کوچک است پس چند بار تکرار لازم است.
- حتی با دادهٔ تکراری، به‌خاطر به‌روزرسانی وزن‌ها، خروجی در epochهای بعدی می‌تواند بهتر شود.
- مدل به‌صورت مستقیم «منطق» را درک نمی‌کند؛ احتمال توکن‌ها را طبق دادهٔ آموزش تنظیم می‌کند.

---

## ۸. AutoGen: تیم‌های چند ایجنت

**AutoGen** فریم‌ورک اوپن‌سورس برای ساخت و مدیریت سیستم‌های **multi-agent** با LLM است: چند ایجنت که با هم حرف می‌زنند، همکاری می‌کنند و وظایف پیچیده را انجام می‌دهند.

**Chat Manager:** خودش یک ایجنت است که گفتگو را هماهنگ می‌کند؛ پیام را به یک ایجنت می‌فرستد، جواب را می‌گیرد و به ایجنت بعدی یا کاربر می‌دهد تا گفتگو تمام شود.

**نمونه پرامپت‌های system_message برای ایجنت‌ها:**

- **CMO (Chief Marketing Officer):**  
  `"You are the Chief Marketing Officer. Your goal is to define high-level marketing strategy and priorities. Consider brand, channels, and budget. Be concise and actionable. When the user or another agent shares a topic, respond with strategic direction and key decisions to make."`

- **Brand Marketer:**  
  `"You are a Brand Marketer. Your goal is to turn the CMO's strategy into concrete campaign ideas and messaging. Suggest specific channels, copy angles, and next steps. Reference what the CMO said and build on it. Keep responses focused and implementable."`

نصب (نمونه):

```bash
pip install -q pyautogen openai google-generativeai "ag2[gemini]"
```

### تنظیم ایجنت‌ها

- **ConversableAgent:** با `system_message` و `llm_config` (مثلاً OpenAI یا Gemini). با `human_input_mode="NEVER"` ورودی مستقیم از کاربر نمی‌گیرد و فقط بر اساس system message و ورودی برنامه پاسخ می‌دهد.
- **UserProxyAgent:** نقش کاربر را بازی می‌کند؛ با `human_input_mode="ALWAYS"` در هر مرحله ورودی از کاربر می‌گیرد. با **code_execution_config** می‌توان اجازهٔ اجرای کد پایتون را داد یا نداد.
- **is_termination_msg:** تابعی که مشخص می‌کند با چه پیامی گفتگو تمام شود (مثلاً "exit" یا "quit").
- **max_consecutive_auto_reply:** حداکثر تعداد پاسخ خودکار پشت‌سرهم بدون ورود کاربر. مثلاً ۱ یعنی بعد از یک پاسخ ایجنت، کنترل به کاربر برگردد تا دوباره چیزی بگوید.

**سؤال: max_consecutive_auto_reply چیست؟**  
تعداد حداکثر replyهایی که همان agent می‌تواند پشت‌سرهم بفرستد بدون اینکه کاربر چیزی بگوید. با ۱، بعد از هر پاسخ ایجنت، نوبت به کاربر (یا manager) می‌رسد. با عدد بالاتر، آن ایجنت می‌تواند چند بار پشت‌سرهم پاسخ دهد (مثلاً برای حل گام‌به‌گام یک مسئله).

### GroupChat و GroupChatManager

```python
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[user_proxy_agent, cmo_agent_gemini, brand_marketer_agent],
    messages=[],   # پیام‌های اولیه (اختیاری)
    max_round=20,  # یک دور = هر ایجنت حداکثر یک بار صحبت کند
)
group_manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
result = group_manager.initiate_chat(recipient=one_of_agents, message="موضوع بحث...")
```

---

## ۹. LangGraph: ورک‌فلوهای ایجنتیک

### ایجنت AI

سیستم‌های خودمختاری که می‌توانند با ابزارهای مختلف و در بازهٔ زمانی طولانی‌تر به‌صورت مستقل کار کنند. هستهٔ آن‌ها معمولاً یک **Augmented LLM** (LLM به‌همراه ابزار و حافظه) است.

### LangGraph

فریم‌ورکی برای مدل‌سازی رفتار ایجنت به‌صورت یک **گراف**: نودها = مراحل (فراخوانی LLM، استفاده از ابزار، تصمیم‌گیری) و یال‌ها = جریان بین مراحل. امکان ساخت ایجنت چندمرحله‌ای، stateful و قابل کنترل را می‌دهد.

**State:** ظرفی است که داده را بین نودها جابه‌جا می‌کند. هر نود state را می‌گیرد و state به‌روزشده برمی‌گرداند.

نمونه نصب:

```bash
pip install langchain langgraph langchain_openai tavily-python amadeus langchain_community graphviz
```

### مثال ساده: دو نود (خلاصه + ترجمه)

با **StateGraph** دو نود تعریف می‌کنید: یکی خلاصه‌سازی، یکی ترجمه. state شامل `input_text`، `summary` و `translated_summary` است. لبه از نود اول به دوم و entry point نود اول است. بعد `compile()` و `invoke(initial_state)`.

**نمونه پرامپت‌ها برای نودها:**

- **خلاصه‌سازی (summarize_step):**  
  `f"Summarize the following text in 3–5 clear bullet points. Preserve the main ideas and key facts. Do not add information that is not in the text.\n\nText:\n{state['input_text']}"`

- **ترجمه (translate_step):**  
  `f"Translate the following summary to Persian (or English, as needed). Keep the same structure (e.g. bullet points). Be accurate and natural.\n\nSummary:\n{state['summary']}"`

برای ایجنت دارای ابزار (مثلاً جستجو)، یک system message مناسب مثال:  
`"You are a helpful assistant with access to a search tool. Use the tool when you need up-to-date or factual information (e.g. current events, definitions). For simple math or general knowledge you already know, answer directly. Always cite what you found when you use search."`

### ایجنت با ابزار و conditional edges

- مدل با **bind_tools** به ابزارها (مثلاً Tavily Search) وصل می‌شود.
- تابع **should_continue** بررسی می‌کند آیا آخرین پیام از نوع AIMessage و دارای **tool_calls** است؛ اگر بله به نود "action" (مثلاً ToolNode) می‌رود، وگرنه به END.
- بعد از اجرای ابزار، خروجی به‌صورت ToolMessage برمی‌گردد و دوباره به نود agent می‌رود.

مدل خودش تصمیم می‌گیرد چه زمانی از ابزار استفاده کند (مثلاً برای ۲+۲ لازم نمی‌بیند جستجو کند).

### ابزار سفارشی

با دکوراتور **`@tool`** از LangChain تابع پایتون را به ابزار تبدیل می‌کنید. مثال جستجوی پرواز با **Amadeus**: پارامترها (مبدا، مقصد، تاریخ، کلاس و ...) را می‌گیرد و با `amadeus_client.shopping.flight_offers_search.get(...)` نتیجه را برمی‌گرداند. این ابزار را مثل Tavily به گراف اضافه می‌کنید و در پرامپت ایجنت توضیح می‌دهید چه زمانی از آن استفاده کند. جستجوی هتل هم در Amadeus وجود دارد ولی محدودتر است.

---

## ۱۰. CrewAI و یادگیری ماشین

قبل از CrewAI، مبحث **Predictive Analytics با Machine Learning** (رگرسیون برای پیش‌بینی عددی) مرور شده است.

### رگرسیون و کتابخانه‌ها

- **pandas** برای بارگذاری و بررسی داده، **sklearn** برای تقسیم داده، imputation، مدل و متریک.
- **SimpleImputer(strategy="median")** برای پر کردن مقادیر خالی با میانه.
- **train_test_split** برای جدا کردن train و test.

### EDA (Exploratory Data Analysis)

۱. **تحلیل متغیر هدف:** توزیع (مثلاً histogram با seaborn)، skew.  
۲. **تحلیل ویژگی‌های عددی:** histogram، boxplot.  
۳. **تحلیل ویژگی‌های دسته‌ای:** countplot با ترتیب فراوانی.  
۴. **رابطه با هدف:** scatter برای عددی–هدف، boxplot برای دسته‌ای–هدف.

### One-Hot Encoding

تبدیل متغیر دسته‌ای به چند ستون دودویی (۰/۱) برای هر دسته. هرگز به‌صورت یک ستون عددی (۱، ۲، ۳ برای قرمز، سبز، آبی) استفاده نکنید؛ مدل آن اعداد را دارای ترتیب و وزن تفسیر می‌کند.

مثال: `pd.get_dummies(..., columns=categorical_cols, drop_first=True)` تا از multicolinearity جلوگیری شود.

### پیش‌پردازش و مدل

- جدا کردن X و y، حذف ستون‌های اضافه، تقسیم train/test.
- **LinearRegression**، **RandomForestRegressor**، **XGBRegressor** با fit و predict.
- متریک‌ها: **MAE**، **MSE**، **RMSE**، **R²**.  
- **Feature importance** در Random Forest: `rf_model.feature_importances_` و مرتب‌سازی.

یک typo در یادداشت اصلی: `features_importances_` باید **`feature_importances_`** باشد؛ و در `sns.scatterplot` و `sns.boxplot` باید ویرگول بین آرگومان‌ها باشد (مثلاً `x="Units Sold", data=df`).

### CrewAI

- **Agent:** نقش، هدف، backstory، LLM و اختیاراً tools.
- **Task:** توضیح کار، خروجی مورد انتظار، ایجنت مسئول و اختیاراً ابزار.
- **Crew:** مجموعه ایجنت‌ها و تسک‌ها با **Process.sequential** (اجرای به‌ترتیب تا تکمیل هر تسک).
- **kickoff()** اجرای crew و **output_log_file** برای ذخیرهٔ لاگ.

پرامپت‌ها (role، goal، backstory، description و expected_output) در CrewAI خیلی مهم هستند و کیفیت خروجی را تعیین می‌کنند.

**نمونه پرامپت‌ها برای ایجنت‌ها و تسک‌های CrewAI (مثلاً تیم تحلیل رگرسیون):**

- **Planner Agent:**  
  `role="Data Science Planner"`, `goal="Define the regression problem, select target variable and key features, and set success criteria"`, `backstory="You are an experienced data scientist who breaks down problems into clear steps."`

- **Analyst/Preprocessor Agent:**  
  `role="Data Analyst"`, `goal="Load and clean the data, perform EDA, handle missing values and encoding, and prepare train/test split"`, `backstory="You are meticulous about data quality and documentation."`

- **Modeler/Evaluator Agent:**  
  `role="ML Engineer"`, `goal="Train and compare regression models (e.g. Linear, Random Forest, XGBoost), report MAE/RMSE/R², and recommend the best model"`, `backstory="You focus on reproducible experiments and clear metrics."`

- **Task (Planning):**  
  `description="Given the dataset description and path, define the regression task: target variable, features to use, and evaluation metrics."`, `expected_output="A short written plan with target, features, and metrics."`

- **Task (Modeling):**  
  `description="Using the preprocessed data, train at least two regression models and compare them. Report MAE, RMSE, R² and feature importance if applicable."`, `expected_output="A summary of model performance and a clear recommendation."`

---

## ۱۱. MCP و OpenAI Agents SDK

### MCP (Model Context Protocol)

پروتکلی اوپن است که به مدل‌های هوش مصنوعی اجازه می‌دهد به‌صورت امن با ابزارها، APIها و داده‌ها ارتباط برقرار کنند؛ شبیه یک درگاه استاندارد (مثل USB-C) برای اپلیکیشن‌های AI.

نصب (نمونه):

```bash
pip install openai-agents gradio openai requests httpx pillow "gradio[mcp]"
```

### سرور MCP با Gradio

یک اپ Gradio با تب‌ها (مثلاً «توضیح مفهوم» و «خلاصهٔ متن») ساخته می‌شود. توابعی مثل `explain_concept` و `summary_text` با **stream=True** از OpenAI پاسخ می‌گیرند و با **yield** تکه‌تکه به UI برمی‌گردانند. در **`launch(..., mcp_server=True)`** همان اپ به‌صورت سرور MCP در دسترس قرار می‌گیرد.

**نمونه پرامپت‌های system برای توابع سرور MCP:**

- **explain_concept (توضیح مفهوم):**  
  `f"You are a patient tutor. Explain the concept or answer the question at exactly this level: {level_desc}. Use the same language as the user (e.g. Persian or English). Do not use jargon above this level. Give one clear, coherent explanation. Use a short example if it helps."`

- **summary_text (خلاصهٔ متن):**  
  `f"You are a summarization assistant. Summarize the following text in a clear and concise way. The summary should be about {int(compression_ratio * 100)}% of the original length. Preserve the main ideas and key facts. Use the same language as the input text. Do not add information that is not in the text."`

در پیام کاربر برای خلاصه باید **متن** (مثلاً `text`) فرستاده شود نه `question`. در بلوک `if __name__ == "__main__"` باید **`demo.launch(...)`** فراخوانی شود (نه `build_demo.launch(...)`). برای استریم از **`yield`** استفاده کنید (نه `yeild`).

### کلاینت MCP

با **MCPServerSse** به آدرس سرور (مثلاً `http://localhost:7860/gradio_api/mcp/sse`) وصل می‌شوید. با جایگزینی `/sse` با `/schema` می‌توان schema ابزارها را با GET گرفت (`fetch_schema`). این schema را می‌توان برای فهم ابزارهای در دسترس استفاده کرد.

در کد نمونه، **`retrieval schema_data`** یک اشتباه نگارشی است و باید **`return schema_data`** باشد.

### OpenAI Agents SDK با MCP

- **Agent** با `name`، `instruction`، `model` و `mcp_servers=[mcp_tool]`.
- **Runner.run** برای اجرای مکالمه؛ ورودی به‌صورت لیست پیام‌ها (مثلاً `[{"role": "user", "content": ...}]`).
- برای مکالمهٔ چندگانه، **result.to_input_list()** را با پیام جدید کاربر جمع می‌کنید و دوباره به **Runner.run** می‌دهید.
- **result.final_output** خروجی نهایی؛ در **to_input_list()** می‌توانید tool callها و arguments را برای لاگ یا دیباگ ببینید.

---

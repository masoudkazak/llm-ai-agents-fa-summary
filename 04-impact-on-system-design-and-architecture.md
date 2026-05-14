# Impact of AI on System Design & Architecture

---

## 1. Modularity

Modularity has always been smart design — but with AI, it becomes **mission critical**.

- Systems must support **frequent change**: swap out logic or components without a full rewrite.
- Use **interfaces** that allow flexibility plus **validation layers** that catch errors before they spread.

---

## 2. Observability

Observability is no longer just about system health or performance. It must cover the **full AI loop**:

- What did the model **receive**?
- What did it **generate**?
- How did that **impact downstream behavior**?

Since AI outputs can vary even with identical inputs, you need:

- Strong **logging**
- **Monitoring**
- End-to-end **traceability**

When something goes wrong, you need to understand not just *what* failed, but *why* the AI chose that path.

---

## 3. Critical Workflow Safeguards

| Safeguard | Purpose |
|---|---|
| Fallback logic | Graceful degradation when AI fails |
| Circuit breakers | Stop cascading failures |
| Retries | Handle transient errors |
| Human review | Catch critical mistakes before they spread |

---

## 4. Decouple Intelligence from Infrastructure

> Don't tightly couple AI logic with your core system.

1. Don't hardcode prompts
2. Don't embed model calls directly in business logic
3. Don't give AI-generated functions direct database access
4. Separate configuration from code

**Pattern:** Wrap model interaction in a dedicated service (e.g., `AIResponderService`) that handles:
- Prompt crafting
- Model communication
- Response validation

---

## 5. Design with Contracts, Not Assumptions

AI outputs are probabilistic — treat them like untrusted external input:

- **Schemas** — define expected output shape
- **Typed APIs** — enforce types at boundaries
- **Validation layers** — reject or flag invalid outputs before they propagate

---

## 6. Feedback & Continuous Improvement Loop

```
Input → AI Model → Prediction → User Action → Response Outcome
              ↑                                        |
              |________________ tuning ________________|
```

Capture:
- What the model **predicted**
- What the **user did**
- Whether the **response worked**

---

## 7. Human vs. AI Responsibilities

| AI (Vibe Coding) | Human Expertise |
|---|---|
| Boilerplate code | Domain logic |
| CRUD operations | Architectural decisions |
| Repetitive syntax | Critical edge cases |

---

## 8. Modular Orchestration

Design each tool or AI component as an **interchangeable service**.

This enables:
- Chaining models
- Calling APIs
- Triggering workflows

...without creating hard dependencies. Benefits:
- Easier to **debug and validate**
- Swap AI models or rewrite prompts without system-wide chaos
- Refactor individual pieces independently

---

## 9. Challenge: Design an AI Daily Assistant Architecture

### Scenario Questions
- What does your architecture look like?
- Where does the AI live in the system?
- What happens when the model needs to be updated?
- How do you keep the human in the loop?

### Think About
- Data flows
- Interfaces
- Logic boundaries
- AI placement
- Infrastructure

### 5 Key Design Questions
1. What is the core function of your system?
2. Which parts will be handled by AI, and which by humans?
3. How will you separate AI from the rest of the system?
4. What happens when the AI fails or gives unexpected output?
5. How will you design for updates — new models or expansion?

### Solution: Layered Architecture

```
User
 ├─ prompt ──────────────────────────────────────────────────────────────┐
 └─ response ←─────────────────────────────────────────────────────┐     │
                                                                   │     ▼
                                                        ┌─────────────────────┐
                                                        │   Orchestration     │
                                                        │  (routing, planning,│
                                                        │   task management)  │
                                                        └─────────┬───────────┘
                                                                  │
                                                        ┌─────────▼───────────┐
                                                        │    Agent Layer      │
                                                        │ (specialized AI &   │
                                                        │  human agents)      │
                                                        └─────────┬───────────┘
                                                                  │
                                                        ┌─────────▼───────────┐
                                                        │  Interface Layer    │
                                                        │ fallback logic,     │
                                                        │ prompt transformer, │
                                                        │ context injector    │
                                                        └──┬──────┬──────┬────┘
                                                           │      │      │
                                              ┌────────────▼┐  ┌──▼──┐ ┌▼──────────────┐
                                              │ Tools Layer │  │Data │ │  LLM Layer    │
                                              │ web search, │  │Store│ │               │
                                              │ API, func   │  └─────┘ └───────────────┘
                                              │ calling     │
                                              └─────────────┘
                                                        +
                                              ┌─────────────────────┐
                                              │  Evaluation Layer   │
                                              │ (quality & safety)  │
                                              └─────────────────────┘
                                                        +
                                              ┌─────────────────────┐
                                              │ Observability Layer  │
                                              └─────────────────────┘
```

---

## 10. Testing AI Systems

Beyond traditional tests, AI systems require:

### Behavior Tests
- Human-in-the-loop review
- Confidence scoring
- Snapshot testing

### Adversarial Testing
Feed the model slightly off-beat inputs to verify graceful handling.

### Security Testing
Verify AI cannot unintentionally **leak sensitive data** or be tricked into **revealing restricted information**.

---

## 11. Challenge: CI/CD Pipeline for AI-Enhanced Systems

### Design Questions
- How will your pipeline handle AI-generated code?
- Will there be pre-commit checks, automated reviews, or tagging strategies?
- Where will human checkpoints be added?
- How will you manage versioning and traceability of AI contributions?
- What's your feedback loop when something goes wrong?

### Test Types
| Test | When |
|---|---|
| Unit & integration tests | Every commit |
| Bias detection | Pre-deploy |
| Prompt stability checks | Pre-deploy |
| Snapshot testing | Pre-deploy |

### Solution: AI-Aware CI/CD Pipeline

```
Commit          → Build              → Test                 → Deploy                → Feedback Loop
─────────────────────────────────────────────────────────────────────────────────────────────────
Human commits     Compile/container    Unit tests             Push to prod            Real-time telemetry
AI commits        Dependencies         Integration tests      Canary/shadow deploy    Drift detection
Pre-commit        AI tagging           AI-specific metrics    Model versioning        Human-in-loop feedback
  checks          Signed builds                               Real-time validation    Prompt refinement
Commit metadata                                               hooks                   Model retraining
AI-specific
  checks
```

---

## 12. Privacy & Compliance (by Design)

Privacy must be **baked into the architecture from day one**.

- **Data minimization** — only collect what's needed
- **Anonymization** — remove identifying information
- **Redaction** — mask sensitive fields before they reach the model

### Access Control
- Set clear boundaries
- Enforce roles
- Maintain audit logs

---

## 13. Challenge: AI Governance Framework

### Design Questions
- Who approves new models or third-party APIs?
- How do you evaluate AI output for accuracy, bias, or harm?
- What's your plan for auditing AI usage over time?
- How do you handle consent and user data rights?
- Who is accountable when things go wrong?

### Solution: RAISE Framework

**Responsible AI Governance and System Engineering**

| Pillar | Focus |
|---|---|
| Responsible AI practices | Ethical guidelines and principles |
| Data privacy & protection | Consent, rights, data handling |
| AI system lifecycle management | From deployment to retirement |
| Algorithm & model oversight | Bias checks, accuracy audits |
| Data governance & quality | Data lineage, integrity |
| Compliance & risk management | Regulatory alignment |
| Approval workflows | Who approves what and when |
| Incident response & feedback | What happens when things go wrong |

---

## 14. Feedback Loops

### Explicit Feedback
- Rating answers
- Suggesting corrections
- Editing AI-generated content

### Implicit Feedback
- Click events
- Response time
- Retries
- Session abandonment

### Route Feedback to Action
- Prompt tuning
- Retraining or fine-tuning
- Threshold adjustment

> Use workflow orchestration tools to automate these steps — but always keep a **human review checkpoint**, especially for safety-critical systems. Apply the **RAISE Framework**.

---

## 15. Resource Optimization

### Right-Size Infrastructure
- Match compute to workload
- Use GPU/TPU only when needed; CPU if sufficient
- Use **auto-scaling** — avoid running idle instances 24/7

### Model Efficiency
- Prefer smaller, more efficient models
- Use **quantization** — shrink models to run faster, use less power, take less space

### Caching & Smart Routing
- Reduces compute and latency
- In RAG systems: lightweight model for common queries, escalate to advanced model only when needed

### Batch & Async Processing
- Batch non-urgent requests to reduce peak load
- Use async workflows to avoid blocking

---

## 16. Challenge: Optimize System Resources

### Checklist
- Could you quantize the model or use a smaller, task-specific version?
- Are you over-provisioning compute for off-peak hours?
- Is there a way to batch requests or cache repeated queries?
- Would moving workload to the edge reduce latency and cloud cost?
- Can you use serverless or auto-scaling instead of always-on infrastructure?
- Have you enabled telemetry to measure usage and detect waste?

### Solution: 5-Step Optimization Plan

**Step 1 — Evaluate what needs to scale**
- Identify compute-heavy processes: training, inference, preprocessing
- Shift to batch where possible
- Cache common queries
- Offload to edge devices

**Step 2 — Choose the right model**
- Use smaller, fine-tuned, or distilled models
- Try zero-shot before training from scratch

**Step 3 — Reuse, recycle, sunset**
- Reuse shared pipelines or embeddings
- Fine-tune existing models for new tasks
- Sunset outdated/unused models
- Free up compute by deprecating old services

**Step 4 — Leverage elastic infrastructure**
- Auto-scaling, serverless, event-driven architecture
- Avoid running VMs 24/7
- Deploy efficient hardware at edge/on-prem (e.g., NVIDIA Jetson, Google Edge TPU)

**Step 5 — Monitor and tune continuously**
- Track CPU/GPU usage, latency, cost per prediction
- Monitor carbon impact (if supported)
- Set alerts for key metrics
- Integrate reviews into ops cycle

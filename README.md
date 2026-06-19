# 🤖 AI Interview Assistant

An AI-powered mock interview tool that reads your resume, generates tailored questions, evaluates your answers, and tracks your improvement over time.

---

## Features

### 🎯 Practice Tab
- Upload your **resume PDF** and generate **3–10 customised questions** at Easy / Medium / Hard difficulty
- **Live answer timer** tracks how long you take
- **Instant AI feedback** with a score out of 10, strengths, weaknesses, and a model answer
- Skip & reset to try different questions

### 📊 Performance Tab *(new)*
- **Score trend chart** across all attempts with a 7/10 target line
- **Distribution breakdown** (Low / Mid / High)
- **AI coaching insights** that adapt to your trend (improving, declining, consistent)
- Full **attempt history** with scores, time taken, and AI feedback
- **Export session** as JSON for offline review

### 💡 Tips & Strategy Tab *(new)*
- Interactive **STAR framework** reference card
- Curated tips for before, during, and after an interview
- **Difficulty guide** explaining what each level tests

---

## Project Structure

```
.
├── app.py                  # Streamlit UI (all tabs)
├── requirements.txt
└── src/
    ├── rag.py              # PDF loading, chunking, vector store, retrieval
    ├── evaluator.py        # Question generation & answer evaluation via Groq
    └── prompts.py          # LangChain prompt templates
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo>
cd ai-interview-assistant
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
streamlit run app.py
```

---

## How It Works

1. **RAG pipeline** — your PDF is chunked and embedded with `sentence-transformers/all-MiniLM-L6-v2`, stored in a local Chroma vector store. The top-4 relevant chunks are retrieved to form the context.
2. **Question generation** — the context is passed to `llama-3.3-70b-versatile` via Groq with a system prompt tuned for technical interviews.
3. **Evaluation** — your answer and the question are sent to the same model which returns a score, strengths, weaknesses, and an improved answer.
4. **Performance analytics** — all scores are stored in Streamlit session state and visualised with Plotly.

---
## For streamlit Deploy
1.add groq api inside settings in secrets
---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| LLM | Groq (llama-3.3-70b-versatile) |
| Orchestration | LangChain |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | Chroma |
| Charts | Plotly |
| PDF parsing | PyPDF |

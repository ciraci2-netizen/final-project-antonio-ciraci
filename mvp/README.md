# Berlin Airbnb Competitive Intelligence Tool
### MVP — Stretch Deliverable

> AI-powered competitive intelligence for independent Airbnb hosts in Berlin.  
> Built with Python · Streamlit · LangChain · GPT-4o-mini · LangSmith

---

## What it does

This tool helps independent Airbnb hosts with 1-3 properties in Berlin make data-driven pricing and positioning decisions. It analyses 9,264 public listings, ranks them using a composite scoring algorithm, and generates 5 actionable business insights via GPT-4o-mini.

**Key features:**
- Composite ranking engine (price competitiveness + review velocity + neighbourhood demand)
- Interactive Streamlit dashboard with neighbourhood filter and KPI metrics
- AI insights referenced to actual data — not generic advice
- Full experiment tracking and traceability via LangSmith
- Graceful error handling on all external calls

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ciraci2-netizen/short-term-rental-ai-analytics
cd short-term-rental-ai-analytics

# 2. Activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r mvp/requirements.txt

# 4. Set up environment variables
cp mvp/.env.example .env
# Edit .env and add your API keys

# 5. Run the app
streamlit run mvp/agent.py
```

Opens at **http://localhost:8501**

---

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| AI inference | OpenAI GPT-4o-mini via LangChain |
| Experiment tracking | LangSmith |
| Data processing | pandas |
| Environment config | python-dotenv |

---

## Project Structure

```
mvp/
├── README.md                   # This file
├── agent.py                    # Main app — ranking engine + Streamlit UI
├── mvp_documentation.md        # Full architecture and documentation
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## Environment Variables

Copy `.env.example` to `.env` in the project root and fill in your keys:

```
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=short-term-rental-analysis
```

**Never commit your `.env` file.**

---

## Data

Uses the [Inside Airbnb Berlin](http://insideairbnb.com/berlin/) public dataset.  
Place `listings.csv` at `data/raw/listings.csv` before running.

---

## Documentation

Full documentation including architecture diagram, ranking engine logic, known limitations, and production roadmap:  
→ [`mvp_documentation.md`](./mvp_documentation.md)

---

*Part of the AI Bootcamp Final Project — Week 9*

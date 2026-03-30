# MVP Documentation
## Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool
**Stretch Deliverable | Version 2.0**
GitHub: `ciraci2-netizen/short-term-rental-ai-analytics`

---

## 1. Overview

This document describes the Working MVP (Stretch Deliverable) of the Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool. The MVP extends the no-code POC into a functional, deployable Python application that a user can interact with to solve a real business problem: helping independent Berlin Airbnb hosts with 1-3 properties make data-driven pricing and positioning decisions.

| Field | Detail |
|---|---|
| Project name | Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool |
| Target user | Independent Airbnb host with 1-3 properties in Berlin |
| Deliverable type | Stretch MVP — functional application beyond the POC |
| MVP version | 2.0 (upgraded from POC-level agent.py) |
| Data source | Inside Airbnb Berlin — public CSV dataset (9,264 listings) |
| GitHub repo | ciraci2-netizen/short-term-rental-ai-analytics |

---

## 2. How the MVP Extends the POC

The POC demonstrated that it is technically feasible to call an LLM with Airbnb listing data and receive structured insights. The MVP goes significantly further on every dimension:

| Feature | POC (n8n) | MVP (agent.py + Streamlit) |
|---|---|---|
| Interface | No UI — n8n workflow only | Interactive Streamlit web app |
| Data scale | 20 rows (hardcoded sample) | 9,264 listings (full Berlin dataset) |
| Ranking | None — raw data passed to LLM | Composite ranking score per listing |
| Filtering | None | Neighbourhood filter + top-N slider |
| KPI dashboard | None | 4 live metrics: listings, median price, avg reviews, top score |
| Prompt quality | Generic — 5 insights from raw data | Data-driven — references ranking scores and neighbourhood median |
| Error handling | Minimal try/except | Graceful failures on all external calls |
| Observability | Basic LangSmith trace | Versioned LangSmith dataset + full trace per run |
| Re-runability | Manual n8n trigger | One-click "Generate Insights" button |

The key instructor feedback has been incorporated: the MVP now enables a host to see how their property compares to similar price points in the same neighbourhood, and surfaces key differentiators (room type, review volume) that could justify a higher asking price.

---

## 3. Architecture Overview

### 3.1 System Diagram

```
[User via Browser]
      ↓  HTTP (localhost:8501)
[Streamlit UI Layer]  ←→  Sidebar filters · KPI metrics · Ranking table · Insights panel
      ↓
[Python Business Logic]  ←→  load_data() · compute_ranking() · analyze_listings()
      ↓                              ↓
[OpenAI GPT-4o-mini]        [LangSmith]
via LangChain ChatOpenAI    Dataset creation + trace logging
      ↓
[Inside Airbnb Berlin CSV]  ←  data/raw/listings.csv (9,264 listings)
```

### 3.2 Core Modules

| Component | Technology | Status | Notes |
|---|---|---|---|
| Data loading | pandas | ✅ Implemented | Price cleaning, null removal, type casting |
| Ranking engine | pandas (custom) | ✅ Implemented | Composite score: price 40% + reviews 40% + demand 20% |
| AI analysis | GPT-4o-mini via LangChain | ✅ Implemented | Neighbourhood-aware prompt, 5 actionable insights |
| Streamlit UI | Streamlit 1.32+ | ✅ Implemented | Filters, KPI row, ranking table, insights panel |
| Experiment tracking | LangSmith | ✅ Implemented | Versioned dataset + automatic trace per run |
| Environment config | python-dotenv | ✅ Implemented | No hardcoded keys, .env.example provided |
| Error handling | try/except | ✅ Implemented | Graceful failure on all external calls |
| CLI mode | Python __main__ | ✅ Implemented | Runs without Streamlit for testing |

---

## 4. Ranking Engine

Each listing receives a composite score between 0 and 1:

| Component | Weight | Logic |
|---|---|---|
| Price competitiveness | 40% | `1 - (price / (neighbourhood_median × 2))`, clipped to [0,1]. Lower price vs median = higher score. |
| Review velocity | 40% | `reviews / max_reviews` in filtered set. High review count = consistent demand signal. |
| Neighbourhood demand | 20% | Listing count per neighbourhood, normalised. More listings = higher demand proxy. |
| **Composite score** | 100% | `(price_score × 0.4) + (review_score × 0.4) + (demand_score × 0.2)` |

The AI prompt explicitly asks GPT-4o-mini to identify room type and feature differentiators (e.g. entire home vs shared room) that could justify a price premium — directly addressing the instructor feedback.

---

## 5. Setup and Installation

### 5.1 Prerequisites

- Python 3.10 or higher
- Git
- OpenAI API key (platform.openai.com)
- LangSmith account and API key (smith.langchain.com)
- Inside Airbnb Berlin `listings.csv` placed at `data/raw/listings.csv`

### 5.2 Installation

```bash
# 1. Clone the repository
git clone https://github.com/ciraci2-netizen/short-term-rental-ai-analytics
cd short-term-rental-ai-analytics

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and add your API keys
```

### 5.3 Environment Variables

Create a `.env` file in the project root (never commit this file):

```
OPENAI_API_KEY=sk-your-openai-key-here
LANGCHAIN_API_KEY=ls__your-langsmith-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=short-term-rental-analysis
```

A `.env.example` template is provided in the repository.

---

## 6. How to Run

### 6.1 Streamlit Web Application (recommended)

```bash
streamlit run agent.py
```

Opens at `http://localhost:8501`. Use the sidebar to filter by neighbourhood and set the number of top listings to analyse. Click **Generate Insights** to run the AI analysis.

### 6.2 CLI Mode (for testing)

```bash
python agent.py
```

Runs the full pipeline in the terminal: loads data, computes ranking, creates LangSmith dataset, prints insights to stdout.

### 6.3 What the User Sees

1. **KPI row** — total listings, median price, average reviews, top ranking score for the selected filter
2. **Ranked table** — top N listings sorted by composite score, with name, neighbourhood, room type, price, reviews, and score
3. **AI insights panel** — 5 actionable insights from GPT-4o-mini, referencing actual data from the ranked listings
4. **LangSmith confirmation** — dataset ID shown after each run for traceability

---

## 7. Error Handling

The MVP is designed to fail gracefully — no unhandled exceptions reach the user interface:

| Scenario | Behaviour |
|---|---|
| `listings.csv` not found | `FileNotFoundError` with clear message; Streamlit shows `st.error()` and stops cleanly |
| Dirty price data | Non-numeric values coerced to NaN and dropped — no crash |
| LangSmith unavailable | Non-blocking `try/except` — warning printed, analysis continues |
| OpenAI API error | Returns user-readable error string instead of raising an exception in the UI |
| Empty filter result | `st.warning()` shown, execution stops before ranking step |

---

## 8. Known Limitations

| Limitation | Detail |
|---|---|
| Dataset freshness | Static CSV snapshot — not live data. Production requires scheduled refresh. |
| Ranking weights | 40/40/20 is heuristic. Production would require calibration via host feedback. |
| No user authentication | Single-user, local only. Production requires auth and hosted deployment. |
| No property input | Host cannot enter their own listing for direct comparison — top planned upgrade. |
| LangSmith deduplication | Each run creates a new timestamped dataset — no deduplication yet. |
| English only | UI and insights in English. Berlin market may prefer German. |
| No persistence | Insights not saved between sessions. Production would store runs in a database. |

---

## 9. What Would Be Needed for Production

- **Live data pipeline** — scheduled refresh of listings data, stored in PostgreSQL
- **Host property input** — form for the host to enter their own listing for direct competitive comparison
- **Authentication** — user login, session management, per-host data isolation
- **Cloud deployment** — Docker container deployed on AWS/GCP/Azure or Streamlit Cloud
- **Prompt evaluation** — LangSmith evaluators to score insight quality over time
- **Multilingual support** — German UI and insights for the Berlin market
- **Calibrated ranking** — host feedback loop to refine composite score weights

---

## 10. Repository Structure

```
short-term-rental-ai-analytics/
├── agent.py                        # Main app — ranking engine, Streamlit UI, CLI mode
├── .env.example                    # Environment variable template (commit this)
├── .env                            # Real API keys (excluded from Git via .gitignore)
├── .gitignore
├── requirements.txt
├── data/
│   └── raw/
│       └── listings.csv            # Inside Airbnb Berlin public dataset
├── compliance/
│   ├── eu_ai_act_compliance.md
│   └── gdpr_documentation.md
├── poc/
│   ├── poc_documentation.md
│   └── poc_workflow.json
├── mvp/
│   └── mvp_documentation.md        # This file
├── use_case_definition.md
├── roi_risk_assessment.md
└── strategic_plan.md
```

---

## 11. MVP Success Criteria

| Criterion | Status |
|---|---|
| Working application — user can interact to solve the stated business problem | ✅ Met |
| Core AI capability demonstrated — not a mockup | ✅ Met |
| Basic error handling — system fails gracefully | ✅ Met |
| Documentation — architecture, setup, run instructions, limitations, production path | ✅ Met |
| Extends the POC — adds ranking, full dataset, Streamlit UI, improved prompt | ✅ Met |
| GitHub repo — organised structure, requirements.txt, .env.example | ✅ Met |
| No hardcoded API keys | ✅ Met |

---

*End of MVP Documentation — Stretch Deliverable*

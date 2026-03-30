# POC Documentation
**Project:** Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool
**Author:** Antonio Ciraci
**Date:** March 2026

---

## 1. Tools Used and Why

| Tool | Purpose | Why Chosen |
|------|---------|------------|
| **n8n** | Workflow orchestration and automation | Open-source, self-hostable, visual interface, supports webhook triggers and HTTP nodes |
| **OpenAI API (GPT-4o-mini)** | AI-generated insights and listing descriptions | Cost-effective, high quality natural language generation |
| **LangSmith** | Monitoring, logging, and experiment tracking | Already integrated in the existing codebase; enables run history and re-run comparison |
| **Python (agent.py)** | Data processing and ranking logic | Pandas-based filtering and scoring of comparable listings |
| **Inside Airbnb Dataset** | Source data for competitive analysis | Publicly available, real Berlin listings data, no personal data concerns |

---

## 2. What the POC Does — Step by Step

### Step 1 — Trigger
An HTTP webhook in n8n receives a POST request containing the host's listing details:
- Listing price
- Neighbourhood group
- Key amenities (e.g. pool, parking, balcony, wifi)
- Number of reviews

### Step 2 — Data Filtering
The n8n workflow calls a Python function node (or HTTP node pointing to the local agent) that:
- Loads the Berlin Airbnb CSV dataset
- Filters listings within the same neighbourhood group
- Filters listings within ±20% of the submitted price point
- Returns the top 10 most comparable listings

### Step 3 — Competitive Ranking
The filtered listings are ranked by a composite score based on:
- Number of reviews (demand signal)
- Price efficiency (price per amenity)
- Review density (reviews per month)

The user's listing is inserted into this ranking to show its relative position.

### Step 4 — Differentiator Identification
The system compares the user's amenities against those of the comparable listings and identifies:
- Features the user has that most comparables lack (key differentiators)
- Features the user is missing that most comparables have (improvement opportunities)

### Step 5 — AI Insight & Description Generation
The ranking data and differentiators are passed to GPT-4o-mini via the OpenAI API with a structured prompt that instructs the model to:
- Generate 3–5 business insights about the listing's competitive position
- Write an optimised listing description that highlights the identified differentiators

### Step 6 — Logging to LangSmith
Every run is traced and logged to LangSmith with:
- Timestamp
- Input data (price, neighbourhood, amenities)
- Generated insights and description
- Ranking position

This enables the host to re-run the analysis at a later date and compare results over time.

### Step 7 — Output
The final output is returned via the n8n webhook response and includes:
- Competitive ranking table (user's listing vs. comparables)
- Key differentiators list
- AI-generated business insights
- Optimised listing description

---

## 3. What AI Capability Is Demonstrated

This POC demonstrates three distinct AI capabilities:

1. **Retrieval and Ranking** — Structured data filtering and scoring using real market data to position a listing competitively. This is a data-driven recommendation system.

2. **Natural Language Generation** — GPT-4o-mini generates human-quality listing descriptions and business insights from structured input data, replacing hours of manual writing.

3. **AI Monitoring and Observability** — LangSmith traces every AI call, enabling evaluation of output quality, detection of regressions, and historical comparison across runs. This is production-grade AI engineering practice.

---

## 4. POC Workflow — n8n Structure

```
[Webhook Trigger]
        ↓
[Set Node — Parse Input]
        ↓
[HTTP Request Node — Call Python Agent]
        ↓
[Function Node — Filter & Rank Comparables]
        ↓
[HTTP Request Node — OpenAI API]
        ↓
[LangSmith Logging Node]
        ↓
[Respond to Webhook — Return Output]
```

> **Note:** Screenshots of the full n8n workflow are available in the `poc_screenshots/` folder.
> **Demo recording:** See link in Section 6 below.

---

## 5. Known Limitations of the POC vs. a Production System

| Limitation | POC State | Production Solution |
|------------|-----------|---------------------|
| **Dataset freshness** | Static CSV snapshot (Inside Airbnb) | Live data pipeline with scheduled updates |
| **Amenity parsing** | Basic string matching | NLP-based amenity extraction from free-text descriptions |
| **User interface** | Webhook input only (JSON) | Full Streamlit or web-based UI |
| **Authentication** | None | User accounts, API key management |
| **Error handling** | Minimal | Full error handling, retries, fallback responses |
| **Scalability** | Single user, local execution | Cloud deployment, multi-user support |
| **Cities** | Berlin only | Multi-city support with city selection |
| **Ranking model** | Simple composite score | ML-based ranking with weighted features |

---

## 6. How to Reproduce / Run the POC

### Prerequisites
- Python 3.10+
- n8n (self-hosted or cloud)
- OpenAI API key
- LangSmith API key
- Berlin Airbnb dataset CSV (download from http://insideairbnb.com/get-the-data/)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/ciraci2-netizen/short-term-rental-ai-analytics
cd short-term-rental-ai-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Add your API keys to .env:
# OPENAI_API_KEY=your_key_here
# LANGCHAIN_API_KEY=your_key_here
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=short-term-rental-analysis

# 4. Place the dataset
# Download listings.csv from Inside Airbnb and place in data/raw/

# 5. Run the agent
python agent.py
```

### n8n Workflow
1. Import `poc_workflow.json` into your n8n instance
2. Configure the OpenAI credentials node with your API key
3. Update the webhook URL to match your n8n instance
4. Activate the workflow
5. Send a POST request to the webhook with the following JSON:

```json
{
  "price": 85,
  "neighbourhood_group": "Mitte",
  "number_of_reviews": 24,
  "amenities": ["wifi", "kitchen", "balcony", "pool"]
}
```

### Expected Output
```json
{
  "ranking_position": 3,
  "total_comparables": 10,
  "differentiators": ["pool", "balcony"],
  "missing_features": [],
  "insights": "Your listing ranks 3rd out of 10 comparable properties...",
  "optimised_description": "Discover this stylish apartment in the heart of Mitte..."
}
```

---

## 7. Demo Recording

> **Link:** *[To be added — screen recording 2–5 minutes showing the POC running end to end]*

The demo covers:
- Submitting a listing via the n8n webhook
- The workflow executing step by step
- The AI-generated output (ranking, differentiators, description)
- The LangSmith run log showing the traced execution

---

*Document version 1.0 — Final Project, AI Bootcamp 2026*

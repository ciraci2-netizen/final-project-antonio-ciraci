# Use Case Definition
**Project:** Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool
**Author:** Antonio Ciraci
**Date:** March 2026

---

## 1. Business Problem Statement

### The Problem
Independent Airbnb hosts in Berlin struggle to understand how their listing performs relative to comparable properties in the market. Without access to competitive intelligence, they are unable to:

- Price their listing competitively based on real market data
- Identify what amenities or features differentiate their property from similar-priced alternatives
- Write compelling listing descriptions that highlight their unique strengths
- Track how their market position changes over time

The result is lost revenue, lower occupancy rates, and missed opportunities to attract guests who would pay a premium for the right property — if only they knew it existed.

### Who Is Affected
**Primary user:** The independent Airbnb host — an individual managing between 1 and 3 properties in Berlin, without access to professional property management tools or data analysts. This person is typically not technically sophisticated. They want clear, actionable answers, not raw data.

### Why This Problem Persists Without AI
Manual competitor research is time-consuming, inconsistent, and quickly becomes outdated. A host would need to manually browse dozens of listings, extract pricing and amenity data, and write their own analysis — a process that takes hours and requires skills most hosts do not have. AI makes this continuous, automated, and accessible to non-technical users.

---

## 2. Company Profile

| Attribute | Detail |
|-----------|--------|
| **Industry** | Short-Term Rental / PropTech |
| **Target User** | Independent Airbnb hosts in Berlin |
| **Company Size** | Individual host or micro-operator (1–3 listings) |
| **Current State** | Hosts rely on intuition, manual browsing, or no competitive research at all |
| **Market Context** | Berlin is one of Europe's largest Airbnb markets with over 14,000 active listings (Inside Airbnb, 2024), making competitive positioning essential but complex |

---

## 3. Proposed AI Solution

### Solution in One Sentence
An AI-powered tool that analyses an Airbnb listing, ranks it against comparable properties in the same price range, identifies key differentiators, and automatically generates an optimised listing description — with every analysis logged for historical comparison.

### What the AI Does
The system performs three core AI functions:

1. **Competitive Ranking** — Given a listing's price point and neighbourhood, the system retrieves comparable listings from the dataset and ranks them by key metrics (reviews, amenities, price-per-feature). The user's listing is positioned within this ranking to show where it stands.

2. **Differentiator Identification** — The AI analyses what features the user's listing has that comparable listings at the same price point do not (e.g. a pool, parking, balcony, or premium location). These become the basis for the optimised description.

3. **Listing Description Generation** — Using the ranking and differentiator data, the AI generates a compelling, optimised listing description that emphasises the property's genuine competitive advantages.

### Type of AI System
- **Retrieval + Ranking:** Structured data filtering and scoring (pandas-based)
- **Natural Language Generation:** GPT-4o-mini via OpenAI API, orchestrated with LangChain
- **Monitoring & Logging:** LangSmith for tracing every analysis run, enabling historical comparison

### End-to-End Workflow

```
User Input (listing details)
        ↓
Load Berlin Airbnb Dataset (Inside Airbnb CSV)
        ↓
Filter comparable listings (same price range ± 20%, same neighbourhood group)
        ↓
Rank user listing vs. comparables (reviews, amenities, price efficiency)
        ↓
Identify differentiators (features user has that comparables lack)
        ↓
Generate optimised listing description (GPT-4o-mini)
        ↓
Log run to LangSmith (timestamp, inputs, outputs, ranking)
        ↓
Display results in Streamlit dashboard
```

---

## 4. Key Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| **Airbnb Host** | Primary user | Increase revenue, improve occupancy, save time |
| **Property Manager** | Secondary user | Manage multiple listings efficiently |
| **AI Developer / Builder** | Solution owner | Deliver a working, compliant, monetisable product |
| **Airbnb Platform** | Indirect | Not directly involved; data sourced from public datasets |
| **Data Regulator (EU)** | Compliance | Ensure GDPR compliance in data processing |

---

## 5. Success Criteria

The solution is considered successful when the following measurable outcomes are achieved:

**Success Criterion 1 — Speed of Insight**
A host receives a competitive ranking report and an optimised listing description within 60 seconds of submitting their listing details.

**Success Criterion 2 — Quality of Ranking**
The competitive ranking correctly identifies at least 5 comparable listings within the same price range (±20%) and the same neighbourhood group, with differentiators explicitly called out.

**Success Criterion 3 — Logging & Re-run Capability**
Every analysis run is logged to LangSmith with a timestamp and retrievable output, enabling the host to compare results across different time periods.

**Success Criterion 4 — User Comprehension**
A non-technical host can interpret the output and take at least one concrete action (e.g. update their listing description) without requiring any explanation or technical support.

---

## 6. Out-of-Scope Boundaries

The following are explicitly outside the scope of this solution:

- **Dynamic pricing automation** — The tool provides insight and recommendations but does not automatically change prices on any platform
- **Direct Airbnb API integration** — All data is sourced from the publicly available Inside Airbnb dataset; no live Airbnb platform connection is made
- **Markets outside Berlin** — The current version is scoped to Berlin listings only
- **Legal or financial advice** — The tool generates data-driven insights; it does not constitute investment, legal, or tax advice
- **Multi-platform analysis** — The tool analyses Airbnb listings only; Booking.com, Vrbo, and other platforms are not included in this version
- **Guest communication automation** — The tool does not generate or send messages to guests

---

## 7. Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Inside Airbnb — Berlin listings dataset |
| **URL** | http://insideairbnb.com/get-the-data/ |
| **Format** | CSV |
| **Key Fields Used** | `price`, `number_of_reviews`, `neighbourhood_group`, `amenities`, `room_type` |
| **Data Type** | Publicly available, non-personal, aggregated listing data |
| **Personal Data** | None — host identifiers are excluded from analysis |

---

*Document version 1.0 — Final Project, AI Bootcamp 2026*

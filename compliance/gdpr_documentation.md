# GDPR Documentation
**Project:** Berlin Airbnb Competitive Intelligence & Listing Optimisation Tool
**Author:** Antonio Ciraci
**Date:** March 2026

---

## 1. Data Flow Map

The following table describes all data that flows through the system:

| Data Type | Source | Processing | Destination | Personal Data? |
|-----------|--------|------------|-------------|----------------|
| Berlin Airbnb listings (price, reviews, neighbourhood, amenities) | Inside Airbnb CSV (public) | Filtering, ranking, scoring | Local memory / pandas | No |
| User-submitted listing details (price, neighbourhood, amenities) | Host via UI | Passed to OpenAI API as prompt input | OpenAI API (transient) | No |
| AI-generated insights and descriptions | OpenAI API | Displayed to user | UI output / LangSmith log | No |
| LangSmith run logs (inputs, outputs, timestamps) | System | Stored in LangSmith cloud | LangSmith (LangChain Inc.) | Potentially — if user inputs contain identifiable info |
| API keys (OpenAI, LangSmith) | Developer configuration | Stored in .env file | Local environment only | Yes — credentials |

### Data Flow Diagram

```
[Host] 
   → submits listing details (price, neighbourhood, amenities)
        ↓
[Streamlit UI / n8n Webhook]
        ↓
[Python Agent — local processing]
   → loads Inside Airbnb CSV (public, no personal data)
   → filters and ranks comparables
        ↓
[OpenAI API — GPT-4o-mini]
   → receives: listing details + comparable data (no personal data)
   → returns: AI-generated insights and description
        ↓
[LangSmith]
   → receives: full run trace (inputs, outputs, timestamps)
   → stores: in LangSmith cloud (US-based, LangChain Inc.)
        ↓
[Host — output displayed in UI]
```

---

## 2. Processing Activities Register

### Activity 1 — Competitive Analysis of Public Listing Data

| Field | Detail |
|-------|--------|
| **Data processed** | Price, number of reviews, neighbourhood group, amenities (from Inside Airbnb CSV) |
| **Purpose** | Filter and rank comparable listings to generate competitive positioning |
| **Legal basis** | Legitimate interest (Art. 6(1)(f)) — publicly available data used for market analysis |
| **Personal data involved** | No — Inside Airbnb data is aggregated and anonymised at source |
| **Retention period** | Not stored — processed in memory during session only |
| **Third-party recipients** | None for this activity |

---

### Activity 2 — AI Insight and Description Generation

| Field | Detail |
|-------|--------|
| **Data processed** | User-submitted listing details (price, neighbourhood, amenities) |
| **Purpose** | Generate competitive insights and optimised listing description |
| **Legal basis** | Legitimate interest (Art. 6(1)(f)) — user voluntarily submits listing data to receive a service |
| **Personal data involved** | Minimal — listing details are property attributes, not personal data. No name, address, or contact information is collected or required |
| **Retention period** | Transient — data is passed to OpenAI API and not stored locally beyond the session |
| **Third-party recipients** | OpenAI (see Section 5) |

---

### Activity 3 — LangSmith Run Logging

| Field | Detail |
|-------|--------|
| **Data processed** | Full run trace: prompt inputs, AI outputs, timestamps, run metadata |
| **Purpose** | Monitoring, evaluation, and historical comparison of AI outputs |
| **Legal basis** | Legitimate interest (Art. 6(1)(f)) — necessary for AI system quality assurance and re-run capability |
| **Personal data involved** | Low risk — inputs contain property attributes only. No personal data is intentionally collected. Risk arises only if a user includes personal information in free-text fields |
| **Retention period** | Retained in LangSmith for the duration of the project / until manually deleted |
| **Third-party recipients** | LangChain Inc. (LangSmith) — see Section 5 |

---

## 3. Data Protection Impact Assessment (DPIA)

**Processing activity assessed:** LangSmith Run Logging (Activity 3 — highest risk activity due to third-party cloud storage)

### 3.1 Description of Processing
Every AI analysis run generates a trace that is automatically sent to LangSmith (operated by LangChain Inc., USA). This trace includes the full prompt sent to GPT-4o-mini (which contains the user's submitted listing details) and the full AI-generated output. Traces are stored in LangSmith's cloud infrastructure and are accessible to the developer via the LangSmith dashboard.

### 3.2 Necessity and Proportionality Assessment

| Question | Assessment |
|----------|------------|
| Is logging necessary for the stated purpose? | Yes — logging is essential for the re-run and historical comparison feature, which is a core requirement of this project |
| Is the scope of data collected proportionate? | Yes — only the data required to reproduce and evaluate a run is logged |
| Could a less privacy-invasive approach achieve the same result? | Partially — local logging (to a file or database) would avoid third-party transfer, but would lose the evaluation and experiment tracking features of LangSmith |

### 3.3 Risks to Data Subjects

| Risk | Likelihood (1–5) | Impact (1–5) | Risk Level |
|------|-----------------|--------------|------------|
| User accidentally includes personal data in listing description field | 2 | 2 | 4 — Low |
| Unauthorised access to LangSmith account exposing run logs | 2 | 3 | 6 — Medium |
| Data transfer to USA without adequate safeguards | 2 | 3 | 6 — Medium |
| LangSmith data breach exposing prompt/output history | 1 | 3 | 3 — Low |

### 3.4 Mitigation Measures

| Risk | Mitigation |
|------|------------|
| Accidental personal data in inputs | Add UI guidance: "Do not include personal information in listing details" |
| Unauthorised LangSmith access | Secure API key storage in .env; restrict LangSmith project access to developer only |
| USA data transfer | Rely on OpenAI and LangChain's Standard Contractual Clauses (SCCs) for EU-US transfers |
| Data breach | Enable LangSmith account 2FA; implement log retention limits |

### 3.5 Residual Risk Rating
**LOW** — The system processes no sensitive personal data. The primary data subjects are the hosts themselves, who voluntarily submit property attributes (not personal information) to receive a service. Residual risks are manageable with the mitigations described above.

---

## 4. Data Subject Rights

The following rights apply under GDPR and are addressed as follows:

| Right | Applicability | How the System Supports It |
|-------|---------------|---------------------------|
| **Right to access (Art. 15)** | Applicable if personal data is logged in LangSmith | Developer can export run logs from LangSmith on request |
| **Right to erasure (Art. 17)** | Applicable for LangSmith logs | Developer can delete specific runs or entire datasets from LangSmith dashboard |
| **Right to rectification (Art. 16)** | Low applicability — no user profiles stored | No user profile data is stored to rectify |
| **Right to data portability (Art. 20)** | Applicable for LangSmith logs | LangSmith supports dataset export in JSON format |
| **Right to object (Art. 21)** | Applicable to legitimate interest processing | User can opt out of LangSmith logging via UI toggle (to be implemented before production) |
| **Right not to be subject to automated decision-making (Art. 22)** | Not applicable | No automated decisions are made about individuals — all outputs are recommendations reviewed by a human |

---

## 5. Third-Party Data Transfers

### 5.1 OpenAI (GPT-4o-mini)

| Field | Detail |
|-------|--------|
| **Data sent** | Listing details submitted by the host (price, neighbourhood, amenities) + comparable listing data from public dataset |
| **Purpose** | Natural language generation of insights and listing descriptions |
| **Legal mechanism** | OpenAI Data Processing Agreement (DPA) + Standard Contractual Clauses (SCCs) for EU-US transfer |
| **Data location** | USA (OpenAI infrastructure) |
| **Retention by OpenAI** | Per OpenAI API data usage policy — API inputs are not used to train models by default (as of 2024) |
| **Personal data transferred** | No — inputs contain property attributes only |

### 5.2 LangChain Inc. (LangSmith)

| Field | Detail |
|-------|--------|
| **Data sent** | Full run traces: prompt inputs, AI outputs, timestamps, run metadata |
| **Purpose** | AI monitoring, evaluation, and historical run comparison |
| **Legal mechanism** | LangSmith Terms of Service + Standard Contractual Clauses (SCCs) for EU-US transfer |
| **Data location** | USA (LangChain Inc. infrastructure) |
| **Retention** | Until manually deleted by developer |
| **Personal data transferred** | Low risk — property attributes only; no intentional personal data transfer |

---

*Document version 1.0 — Final Project, AI Bootcamp 2026*

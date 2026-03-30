from dotenv import load_dotenv
import os
import pandas as pd
import warnings
from datetime import datetime

# Load environment variables (local dev)
load_dotenv()

# Streamlit Cloud secrets override
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
except Exception:
    pass

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "short-term-rental-analysis"

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import Client

# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------

def load_data(path="mvp/data/raw/listings.csv"):
    """Load and clean the Airbnb listings dataset."""
    try:
        df = pd.read_csv(path)
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.dropna(subset=["price", "number_of_reviews", "neighbourhood_group"])
        df["number_of_reviews"] = df["number_of_reviews"].astype(int)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}. Make sure listings.csv is in data/raw/")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


# -------------------------------------------------------
# RANKING ENGINE
# -------------------------------------------------------

def compute_ranking(df, neighbourhood_filter=None):
    working = df.copy()
    if neighbourhood_filter and neighbourhood_filter != "All":
        working = working[working["neighbourhood_group"] == neighbourhood_filter]
    if working.empty:
        return working
    neighbourhood_median = working.groupby("neighbourhood_group")["price"].transform("median")
    working["price_score"] = 1 - (working["price"] / (neighbourhood_median * 2)).clip(0, 1)
    max_reviews = working["number_of_reviews"].max()
    working["review_score"] = working["number_of_reviews"] / max_reviews if max_reviews > 0 else 0
    neighbourhood_counts = working["neighbourhood_group"].map(
        working["neighbourhood_group"].value_counts()
    )
    working["demand_score"] = neighbourhood_counts / neighbourhood_counts.max()
    working["ranking_score"] = (
        working["price_score"] * 0.4 +
        working["review_score"] * 0.4 +
        working["demand_score"] * 0.2
    ).round(3)
    return working.sort_values("ranking_score", ascending=False)


# -------------------------------------------------------
# LANGSMITH DATASET
# -------------------------------------------------------

def create_langsmith_dataset(sample):
    try:
        client = Client()
        dataset_name = f"airbnb_listings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Berlin Airbnb listings — ranked sample for AI analysis"
        )
        for _, row in sample.iterrows():
            client.create_example(
                dataset_id=dataset.id,
                inputs={
                    "name": str(row.get("name", "N/A")),
                    "price": float(row["price"]),
                    "number_of_reviews": int(row["number_of_reviews"]),
                    "neighbourhood_group": str(row["neighbourhood_group"]),
                    "room_type": str(row.get("room_type", "N/A")),
                    "ranking_score": float(row.get("ranking_score", 0)),
                },
                outputs={"insight_type": "competitive_ranking"}
            )
        return dataset.id
    except Exception as e:
        print(f"LangSmith dataset error (non-blocking): {e}")
        return None


# -------------------------------------------------------
# AI ANALYSIS
# -------------------------------------------------------

def analyze_listings(sample_df, neighbourhood_filter="All"):
    top = sample_df.head(10)
    cols = ["name", "price", "number_of_reviews", "neighbourhood_group", "room_type", "ranking_score"]
    available_cols = [c for c in cols if c in top.columns]
    data_text = top[available_cols].to_string(index=False)
    neighbourhood_context = (
        f"Focus area: {neighbourhood_filter} neighbourhood."
        if neighbourhood_filter != "All"
        else "Analysis covers all Berlin neighbourhoods."
    )
    prompt = f"""
You are a competitive intelligence analyst specialised in the Berlin short-term rental market.

{neighbourhood_context}

Below are the top 10 ranked Airbnb listings based on a composite score combining price competitiveness, review velocity, and neighbourhood demand.

{data_text}

Generate 5 actionable business insights for an independent Airbnb host with 1-3 properties. Focus on:
1. Pricing positioning vs the neighbourhood median
2. Demand signals from review counts
3. Key differentiators that could justify a higher price (room type, location)
4. Risks or opportunities in the current market
5. One concrete recommendation the host should act on this week

Be specific, data-driven, and avoid generic advice.
"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"AI analysis error: {e}"


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="BerlinHostAIQ — Competitive Intelligence for Airbnb Hosts",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

        :root {
            --bg-primary: #0A1628;
            --bg-secondary: #0F1F38;
            --bg-card: #132140;
            --cyan: #00D4FF;
            --gold: #FFD700;
            --white: #FFFFFF;
            --muted: #8899AA;
            --border: rgba(0, 212, 255, 0.15);
        }

        /* Remove Streamlit default white padding/margin */
        .stApp { background-color: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        header[data-testid="stHeader"] { background-color: var(--bg-primary) !important; }

        [data-testid="stSidebar"] {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--white) !important; }

        /* Dropdown options readability */
        [data-baseweb="select"] * { color: var(--white) !important; background-color: var(--bg-card) !important; }
        [data-baseweb="popover"] { background-color: var(--bg-card) !important; }
        [data-baseweb="menu"] { background-color: var(--bg-card) !important; }
        [data-baseweb="menu"] li { color: #E0EEFF !important; background-color: var(--bg-card) !important; }
        [data-baseweb="menu"] li:hover { background-color: rgba(0,212,255,0.15) !important; color: var(--cyan) !important; }

        .hero-header { padding: 1.5rem 0 1.5rem 0; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
        .hero-title { font-family: 'Space Mono', monospace; font-size: 2.4rem; font-weight: 700; color: var(--white); letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
        .hero-title span { color: var(--cyan); }
        .hero-subtitle { font-size: 0.95rem; color: var(--muted); margin-top: 0.5rem; font-weight: 300; letter-spacing: 0.05em; }
        .hero-tag { display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: var(--cyan); font-size: 0.7rem; font-family: 'Space Mono', monospace; padding: 3px 10px; border-radius: 20px; margin-right: 6px; letter-spacing: 0.08em; }

        .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
        .kpi-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.5rem; position: relative; overflow: hidden; }
        .kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--cyan), transparent); }
        .kpi-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; font-family: 'Space Mono', monospace; margin-bottom: 0.4rem; }
        .kpi-value { font-size: 2rem; font-weight: 600; color: var(--white); line-height: 1; }
        .kpi-value.cyan { color: var(--cyan); }
        .kpi-value.gold { color: var(--gold); }

        /* Section header — refined, not all-caps mono */
        .section-header { font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 600; color: var(--cyan); letter-spacing: 0.05em; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px; }
        .section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

        [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden; }

        .stButton > button { background: linear-gradient(135deg, var(--cyan), #0099CC) !important; color: #0A1628 !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.08em !important; border: none !important; border-radius: 8px !important; padding: 0.7rem 2rem !important; }
        .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important; }

        .insights-box { background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--cyan); border-radius: 0 12px 12px 0; padding: 1.5rem 2rem; margin-top: 1rem; color: #C8D8E8; line-height: 1.8; font-size: 0.95rem; }

        .footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); font-size: 0.75rem; color: var(--muted); font-family: 'Space Mono', monospace; letter-spacing: 0.05em; }

        p, li, span, label { color: #C8D8E8 !important; }
        h1, h2, h3 { color: var(--white) !important; }
    </style>
    """, unsafe_allow_html=True)

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">Berlin<span>Host</span>AIQ</p>
        <p class="hero-subtitle">AI-Powered Competitive Intelligence for Independent Airbnb Hosts</p>
        <div style="margin-top: 0.8rem;">
            <span class="hero-tag">GPT-4o-mini</span>
            <span class="hero-tag">LangSmith</span>
            <span class="hero-tag">9,264 Berlin Listings</span>
            <span class="hero-tag">Real-time Ranking</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0 0.5rem; border-bottom: 1px solid rgba(0,212,255,0.15); margin-bottom: 1rem;">
            <p style="font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00D4FF; letter-spacing: 0.15em; text-transform: uppercase; margin:0;">Filters</p>
        </div>
        """, unsafe_allow_html=True)

        neighbourhoods = ["All"] + sorted(df["neighbourhood_group"].dropna().unique().tolist())
        selected_neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
        top_n = st.slider("Top listings to analyse", min_value=5, max_value=50, value=10)

        st.markdown(f"""
        <div style="margin-top: 2rem; border-top: 1px solid rgba(0,212,255,0.15); padding-top: 1rem;">
        <p style="font-size: 0.75rem; color: #8899AA; font-family: 'Space Mono', monospace; line-height: 1.8;">
        DATASET<br>
        <span style="color: #00D4FF; font-size: 1rem;">{len(df):,}</span> listings loaded<br>
        Source: Inside Airbnb Berlin
        </p>
        <p style="font-size: 0.7rem; color: #8899AA; font-family: 'Space Mono', monospace; margin-top: 1rem; line-height: 1.8;">
        RANKING FORMULA<br>
        Price 40% · Reviews 40%<br>Demand 20%
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Compute ranking
    ranked_df = compute_ranking(df, selected_neighbourhood)
    if ranked_df.empty:
        st.warning("No listings found for the selected filter.")
        st.stop()

    filtered = ranked_df if selected_neighbourhood == "All" else ranked_df[ranked_df["neighbourhood_group"] == selected_neighbourhood]

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Total Listings</div>
            <div class="kpi-value cyan">{len(filtered):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Median Price</div>
            <div class="kpi-value">&#8364;{filtered['price'].median():.0f}<span style="font-size:1rem;color:#8899AA">/night</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Reviews</div>
            <div class="kpi-value">{filtered['number_of_reviews'].mean():.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Top Score</div>
            <div class="kpi-value gold">{filtered['ranking_score'].max():.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rankings Table
    st.markdown('<div class="section-header">Top Ranked Listings</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["name", "neighbourhood_group", "room_type", "price", "number_of_reviews", "ranking_score"] if c in ranked_df.columns]
    st.dataframe(
        ranked_df[display_cols].head(top_n).reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    # AI Insights
    st.markdown('<div class="section-header">AI Competitive Insights</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("Generate Insights →", type="primary")

    if run_button:
        with st.spinner("Analysing market data with GPT-4o-mini..."):
            sample = ranked_df.head(top_n)
            dataset_id = create_langsmith_dataset(sample)
            if dataset_id:
                st.caption(f"✓ Logged to LangSmith · dataset: {dataset_id}")
            insights = analyze_listings(ranked_df, selected_neighbourhood)
        st.markdown(f'<div class="insights-box">{insights}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        BerlinHostAIQ &nbsp;·&nbsp; Antonio Ciraci &nbsp;·&nbsp; AI Bootcamp Final Project &nbsp;·&nbsp; March 2026
        &nbsp;·&nbsp; Data: Inside Airbnb (public)
        &nbsp;·&nbsp; Model: GPT-4o-mini
        &nbsp;·&nbsp; Tracing: LangSmith
        &nbsp;·&nbsp; <a href="https://github.com/ciraci2-netizen/final-project-antonio-ciraci" style="color: #00D4FF; text-decoration: none;">GitHub ↗</a>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        print("Loading dataset...")
        df = load_data()
        print(f"✓ {len(df):,} listings loaded")
        print("\nComputing ranking...")
        ranked = compute_ranking(df)
        print(f"✓ Top listing: {ranked.iloc[0].get('name', 'N/A')} — score: {ranked.iloc[0]['ranking_score']}")
        print("\nCreating LangSmith dataset...")
        create_langsmith_dataset(ranked.head(20))
        print("\nGenerating AI insights...")
        insights = analyze_listings(ranked)
        print("\n" + "=" * 60)
        print("COMPETITIVE INTELLIGENCE REPORT")
        print("=" * 60)
        print(insights)

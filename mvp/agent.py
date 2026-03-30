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

def load_data(path="data/raw/listings.csv"):
    """Load and clean the Airbnb listings dataset."""
    try:
        df = pd.read_csv(path)
        # Clean price column (remove $ and commas if present)
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
    """
    Score each listing on a composite index:
      - Price competitiveness (lower price vs neighbourhood median = higher score)
      - Review velocity (more reviews = higher demand signal)
      - Neighbourhood demand (neighbourhoods with more listings = higher demand)

    Returns a dataframe with a ranking_score column, sorted descending.
    """
    working = df.copy()

    if neighbourhood_filter and neighbourhood_filter != "All":
        working = working[working["neighbourhood_group"] == neighbourhood_filter]

    if working.empty:
        return working

    # 1. Price competitiveness score (0-1): cheaper vs neighbourhood median = better
    neighbourhood_median = working.groupby("neighbourhood_group")["price"].transform("median")
    working["price_score"] = 1 - (working["price"] / (neighbourhood_median * 2)).clip(0, 1)

    # 2. Review velocity score (0-1): normalised number of reviews
    max_reviews = working["number_of_reviews"].max()
    working["review_score"] = working["number_of_reviews"] / max_reviews if max_reviews > 0 else 0

    # 3. Neighbourhood demand score (0-1): based on listing count per neighbourhood
    neighbourhood_counts = working["neighbourhood_group"].map(
        working["neighbourhood_group"].value_counts()
    )
    working["demand_score"] = neighbourhood_counts / neighbourhood_counts.max()

    # Composite score (weighted)
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
    """Create a versioned LangSmith dataset from the sample."""
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
        print(f"✓ LangSmith dataset created: {dataset_name}")
        return dataset.id
    except Exception as e:
        print(f"⚠️  LangSmith dataset error (non-blocking): {e}")
        return None


# -------------------------------------------------------
# AI ANALYSIS
# -------------------------------------------------------

def analyze_listings(sample_df, neighbourhood_filter="All"):
    """
    Generate AI-powered competitive insights from the top-ranked listings.
    Returns the model response as a string.
    """
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
        return f"⚠️ AI analysis error: {e}"


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Berlin Airbnb Intelligence Tool",
        page_icon="🏠",
        layout="wide"
    )

    st.title("🏠 Berlin Airbnb Competitive Intelligence Tool")
    st.caption("AI-powered insights for independent hosts · Powered by GPT-4o-mini + LangSmith")

    # --- Load data ---
    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # --- Sidebar filters ---
    st.sidebar.header("Filters")
    neighbourhoods = ["All"] + sorted(df["neighbourhood_group"].dropna().unique().tolist())
    selected_neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhoods)

    top_n = st.sidebar.slider("Top listings to analyse", min_value=5, max_value=50, value=10)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset:** {len(df):,} listings loaded")
    st.sidebar.markdown(f"**Source:** Inside Airbnb Berlin")

    # --- Compute ranking ---
    ranked_df = compute_ranking(df, selected_neighbourhood)

    if ranked_df.empty:
        st.warning("No listings found for the selected filter.")
        st.stop()

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)
    filtered = ranked_df if selected_neighbourhood == "All" else ranked_df[ranked_df["neighbourhood_group"] == selected_neighbourhood]

    col1.metric("Total Listings", f"{len(filtered):,}")
    col2.metric("Median Price", f"€{filtered['price'].median():.0f}/night")
    col3.metric("Avg Reviews", f"{filtered['number_of_reviews'].mean():.0f}")
    col4.metric("Top Score", f"{filtered['ranking_score'].max():.3f}")

    st.markdown("---")

    # --- Ranking table ---
    st.subheader("📊 Top Ranked Listings")

    display_cols = [c for c in ["name", "neighbourhood_group", "room_type", "price", "number_of_reviews", "ranking_score"] if c in ranked_df.columns]
    st.dataframe(
        ranked_df[display_cols].head(top_n).reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")

    # --- AI Analysis ---
    st.subheader("🤖 AI Competitive Insights")

    if st.button("Generate Insights", type="primary"):
        with st.spinner("Analysing top listings with GPT-4o-mini..."):
            sample = ranked_df.head(top_n)

            # Log to LangSmith
            dataset_id = create_langsmith_dataset(sample)
            if dataset_id:
                st.caption(f"✓ Logged to LangSmith — dataset ID: {dataset_id}")

            # Generate insights
            insights = analyze_listings(ranked_df, selected_neighbourhood)

        st.markdown(insights)

    st.markdown("---")
    st.caption("Data: Inside Airbnb (public dataset) · Model: GPT-4o-mini · Tracing: LangSmith")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        # CLI mode: run analysis directly
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
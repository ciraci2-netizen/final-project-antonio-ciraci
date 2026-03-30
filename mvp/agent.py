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

        .stApp { background-color: var(--bg-primary); font-family: 'DM Sans', sans-serif; }
        .block-container { padding-top: 1rem !important; }
        header[data-testid="stHeader"] { background-color: var(--bg-primary) !important; }

        [data-testid="stSidebar"] { background-color: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] * { color: var(--white) !important; }

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

        .section-header { font-family: 'DM Sans', sans-serif; font-size: 1rem; font-weight: 600; color: var(--cyan); letter-spacing: 0.05em; margin: 2rem 0 1rem 0; display: flex; align-items: center; gap: 10px; }
        .section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

        [data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden; }

        .stButton > button { background: linear-gradient(135deg, var(--cyan), #0099CC) !important; color: #0A1628 !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.08em !important; border: none !important; border-radius: 8px !important; padding: 0.7rem 2rem !important; }
        .stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important; }

        .insights-box { background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--cyan); border-radius: 0 12px 12px 0; padding: 1.5rem 2rem; margin-top: 1rem; color: #C8D8E8; line-height: 1.8; font-size: 0.95rem; white-space: pre-wrap; }

        .footer { margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border); font-size: 0.75rem; color: var(--muted); font-family: 'Space Mono', monospace; letter-spacing: 0.05em; }

        p, li, span, label { color: #C8D8E8 !important; }
        h1, h2, h3 { color: var(--white) !important; }

        /* Selectbox dropdown fix */
        div[data-baseweb="select"] > div {
            background-color: #132140 !important;
            border-color: rgba(0,212,255,0.3) !important;
            color: #E0EEFF !important;
        }
        div[data-baseweb="select"] span { color: #E0EEFF !important; }
        div[data-baseweb="select"] svg { fill: #00D4FF !important; }
        div[data-baseweb="popover"] div[role="listbox"] {
            background-color: #0F1F38 !important;
            border: 1px solid rgba(0,212,255,0.2) !important;
        }
        div[data-baseweb="popover"] li {
            background-color: #0F1F38 !important;
            color: #E0EEFF !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: rgba(0,212,255,0.15) !important;
            color: #00D4FF !important;
        }
        div[data-baseweb="popover"] li[aria-selected="true"] {
            background-color: rgba(0,212,255,0.2) !important;
            color: #00D4FF !important;
        }
        div[data-baseweb="input"] input {
            background-color: #132140 !important;
            color: #E0EEFF !important;
            border-color: rgba(0,212,255,0.3) !important;
        }
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
    # Language
    LANG = {
        "EN": {
            "filters": "Filters", "neighbourhood": "Neighbourhood", "room_type": "Room type",
            "top_n": "Top listings to analyse", "dataset": "DATASET", "formula": "RANKING FORMULA",
            "rankings": T["rankings"], "caption": T["caption"],
            "map_title": "Berlin Neighbourhoods — Price & Demand", "chart_title": "Median Price by Neighbourhood",
            "insights_title": "AI Competitive Insights", "generate": T["generate"],
            "analyse_title": "Analyse My Listing", "how_it_works": T["how_it_works"],
            "how_text": "Enter your listing details — BerlinHostAIQ will rank you against competitors and generate personalised AI recommendations.",
            "price_label": T["price_label"], "nb_label": T["nb_label"],
            "reviews_label": T["reviews_label"], "amenities_label": T["amenities_label"],
            "submit": "Analyse My Listing →", "your_score": T["your_score"], T["beats"]: "beats",
            "listings_label": "listings", "breakdown": T["breakdown"],
            "total": "Total Listings", "median": "Median Price", "avg": "Avg Reviews", "top": "Top Score",
            "spinner_market": T["spinner_market"],
            "spinner_listing": T["spinner_listing"],
            "toast_market": T["toast_market"], "toast_listing": T["toast_listing"],
        },
        "IT": {
            "filters": "Filtri", "neighbourhood": "Quartiere", "room_type": "Tipo di stanza",
            "top_n": "Top listing da analizzare", "dataset": "DATASET", "formula": "FORMULA RANKING",
            "rankings": "Top Listing per Ranking", "caption": "Score = Competitivitá prezzo (40%) + Velocitá recensioni (40%) + Domanda quartiere (20%)",
            "map_title": "Quartieri di Berlino — Prezzi & Domanda", "chart_title": "Prezzo Mediano per Quartiere",
            "insights_title": "AI Competitive Insights", "generate": "Genera Insights →",
            "analyse_title": "Analizza il mio Listing", "how_it_works": "COME FUNZIONA",
            "how_text": "Inserisci i dati del tuo listing — BerlinHostAIQ ti posizionerá vs i competitor e genererá raccomandazioni personalizzate.",
            "price_label": "Prezzo a notte (€)", "nb_label": "Il tuo quartiere",
            "reviews_label": "Numero di recensioni", "amenities_label": "Amenities principali (es. balcone, giardino, parcheggio, wifi)",
            "submit": "Analizza il mio Listing →", "your_score": "IL TUO SCORE", "beats": "supera",
            "listings_label": "listing", "breakdown": "DETTAGLIO SCORE",
            "total": "Listing Totali", "median": "Prezzo Mediano", "avg": "Recensioni Medie", "top": "Top Score",
            "spinner_market": "Analisi dati di mercato con GPT-4o-mini...",
            "spinner_listing": "Posizionamento del tuo listing vs competitor...",
            "toast_market": "✓ Analisi completata!", "toast_listing": "✓ Il tuo listing é stato analizzato!",
        },
        "DE": {
            "filters": "Filter", "neighbourhood": "Bezirk", "room_type": "Zimmertyp",
            "top_n": "Top-Inserate analysieren", "dataset": "DATENSATZ", "formula": "RANKING-FORMEL",
            "rankings": "Top-Inserate nach Ranking", "caption": "Score = Preiswettbewerb (40%) + Bewertungsgeschwindigkeit (40%) + Bezirksnachfrage (20%)",
            "map_title": "Berliner Bezirke — Preise & Nachfrage", "chart_title": "Medianpreis nach Bezirk",
            "insights_title": "KI-Wettbewerbsanalyse", "generate": "Insights generieren →",
            "analyse_title": "Mein Inserat analysieren", "how_it_works": "SO FUNKTIONIERT ES",
            "how_text": "Geben Sie Ihre Inserat-Details ein — BerlinHostAIQ vergleicht Sie mit Mitbewerbern und erstellt personalisierte Empfehlungen.",
            "price_label": "Ihr Nachtpreis (€)", "nb_label": "Ihr Bezirk",
            "reviews_label": "Anzahl Bewertungen", "amenities_label": "Hauptausstattung (z.B. Balkon, Garten, Parkplatz, WLAN)",
            "submit": "Inserat analysieren →", "your_score": "IHR SCORE", "beats": "besser als",
            "listings_label": "Inserate", "breakdown": "SCORE-AUFSCHLÜSSELUNG",
            "total": "Inserate gesamt", "median": "Medianpreis", "avg": "Ø Bewertungen", "top": "Top Score",
            "spinner_market": "Marktdaten werden mit GPT-4o-mini analysiert...",
            "spinner_listing": "Ihr Inserat wird mit Mitbewerbern verglichen...",
            "toast_market": "✓ Analyse abgeschlossen!", "toast_listing": "✓ Ihr Inserat wurde analysiert!",
        },
    }

    with st.sidebar:
        lang_choice = st.radio("🌐 Language", ["EN", "IT", "DE"], horizontal=True)
        T = LANG[lang_choice]

        st.markdown(f"""
        <div style="padding: 1rem 0 0.5rem; border-bottom: 1px solid rgba(0,212,255,0.15); margin-bottom: 1rem; margin-top: 0.5rem;">
            <p style="font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #00D4FF; letter-spacing: 0.15em; text-transform: uppercase; margin:0;">{T["filters"]}</p>
        </div>
        """, unsafe_allow_html=True)

        neighbourhoods = ["All"] + sorted(df["neighbourhood_group"].dropna().unique().tolist())
        selected_neighbourhood = st.selectbox(T["neighbourhood"], neighbourhoods)
        room_types = ["All"] + sorted(df["room_type"].dropna().unique().tolist())
        selected_room_type = st.selectbox(T["room_type"], room_types)
        top_n = st.slider(T["top_n"], min_value=5, max_value=50, value=10)

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

    # Apply filters
    df_filtered = df.copy()
    if selected_room_type != "All":
        df_filtered = df_filtered[df_filtered["room_type"] == selected_room_type]

    ranked_df = compute_ranking(df_filtered, selected_neighbourhood)
    if ranked_df.empty:
        st.warning("No listings found for the selected filter.")
        st.stop()

    filtered = ranked_df if selected_neighbourhood == "All" else ranked_df[ranked_df["neighbourhood_group"] == selected_neighbourhood]

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">{T["total"]}</div>
            <div class="kpi-value cyan">{len(filtered):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{T["median"]}</div>
            <div class="kpi-value">&#8364;{filtered['price'].median():.0f}<span style="font-size:1rem;color:#8899AA">/night</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{T["avg"]}</div>
            <div class="kpi-value">{filtered['number_of_reviews'].mean():.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{T["top"]}</div>
            <div class="kpi-value gold">{filtered['ranking_score'].max():.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rankings Table
    st.markdown('<div class="section-header">Top Ranked Listings</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["name", "neighbourhood_group", "room_type", "price", "number_of_reviews", "ranking_score"] if c in ranked_df.columns]
    st.dataframe(
        ranked_df[display_cols].head(top_n).reset_index(drop=True).rename(index=lambda x: x + 1),
        use_container_width=True,
        height=400
    )
    st.caption("Score = Price competitiveness (40%) + Review velocity (40%) + Neighbourhood demand (20%)")

    # Map + Chart
    col_map, col_chart = st.columns([3, 2])

    with col_map:
        st.markdown(f'<div class="section-header">{T["map_title"]}</div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            map_data = (
                df.dropna(subset=["latitude", "longitude", "neighbourhood_group"])
                .groupby("neighbourhood_group")
                .agg(
                    lat=("latitude", "median"),
                    lon=("longitude", "median"),
                    median_price=("price", "median"),
                    listing_count=("price", "count"),
                    avg_reviews=("number_of_reviews", "mean"),
                )
                .reset_index()
            )
            fig_map = px.scatter_mapbox(
                map_data,
                lat="lat",
                lon="lon",
                size="listing_count",
                color="median_price",
                hover_name="neighbourhood_group",
                hover_data={"median_price": ":€.0f", "listing_count": True, "lat": False, "lon": False},
                color_continuous_scale=[[0, "#0D2137"], [0.4, "#0099CC"], [1, "#00D4FF"]],
                size_max=50,
                zoom=10,
                center={"lat": 52.52, "lon": 13.405},
                mapbox_style="carto-darkmatter",
                labels={"median_price": "Median Price (€)", "listing_count": "Listings"},
            )
            fig_map.update_layout(
                paper_bgcolor="#0A1628",
                margin=dict(l=0, r=0, t=0, b=0),
                height=430,
                coloraxis_colorbar=dict(
                    title="Price (€)",
                    tickfont=dict(color="#8899AA", size=10),
                    bgcolor="#0A1628",
                    bordercolor="rgba(0,212,255,0.2)",
                ),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.caption(f"Map unavailable: {e}")

    with col_chart:
        st.markdown(f'<div class="section-header">{T["chart_title"]}</div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            price_by_nb = (
                df.groupby("neighbourhood_group")["price"]
                .median()
                .reset_index()
                .sort_values("price", ascending=True)
                .rename(columns={"neighbourhood_group": "Neighbourhood", "price": "Median Price (€)"})
            )
            fig = px.bar(
                price_by_nb,
                x="Median Price (€)",
                y="Neighbourhood",
                orientation="h",
                color="Median Price (€)",
                color_continuous_scale=[[0, "#132140"], [0.5, "#00A0CC"], [1, "#00D4FF"]],
            )
            fig.update_layout(
                paper_bgcolor="#0A1628",
                plot_bgcolor="#0A1628",
                font_color="#C8D8E8",
                font_family="DM Sans",
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#8899AA"),
                yaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#C8D8E8"),
                height=430,
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")

    # AI Market Insights
    st.markdown(f'<div class="section-header">{T["insights_title"]}</div>', unsafe_allow_html=True)
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
        st.toast("✓ Analysis complete!", icon="🤖")

    # Analyse My Listing
    st.markdown(f'<div class="section-header">{T["analyse_title"]}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #132140; border: 1px solid rgba(0,212,255,0.15); border-left: 3px solid #FFD700; border-radius: 0 12px 12px 0; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
        <p style="color: #FFD700; font-size: 0.8rem; font-family: Space Mono, monospace; letter-spacing: 0.1em; margin: 0 0 0.3rem;">HOW IT WORKS</p>
        <p style="color: #C8D8E8; font-size: 0.9rem; margin: 0;">Enter your listing details — BerlinHostAIQ will rank you against competitors and generate personalised AI recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("listing_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            host_price = st.number_input("Your nightly price (€)", min_value=10, max_value=1000, value=85)
        with fc2:
            host_neighbourhood = st.selectbox("Your neighbourhood", ["All"] + sorted(df["neighbourhood_group"].dropna().unique().tolist()), key="host_nb")
        with fc3:
            host_reviews = st.number_input("Number of reviews", min_value=0, max_value=5000, value=25)
        host_amenities = st.text_input("Key amenities (e.g. balcony, garden, parking, wifi)", value="balcony, wifi, fully equipped kitchen")
        submitted = st.form_submit_button(T["submit"], type="primary")

    if submitted:
        with st.spinner("Ranking your listing vs competitors..."):
            comp_df = compute_ranking(df, host_neighbourhood if host_neighbourhood != "All" else None)
            nb_median = comp_df["price"].median()
            nb_listings = len(comp_df)
            price_score = round(1 - min(host_price / (nb_median * 2), 1), 3)
            max_reviews = comp_df["number_of_reviews"].max()
            review_score = round(host_reviews / max_reviews if max_reviews > 0 else 0, 3)
            demand_score = round(nb_listings / df["neighbourhood_group"].value_counts().max(), 3)
            my_score = round(price_score * 0.4 + review_score * 0.4 + demand_score * 0.2, 3)
            percentile = round((comp_df["ranking_score"] < my_score).mean() * 100, 1)

            personal_prompt = f"""
You are a competitive intelligence analyst for the Berlin Airbnb market.

A host has provided their listing details:
- Nightly price: €{host_price}
- Neighbourhood: {host_neighbourhood}
- Number of reviews: {host_reviews}
- Key amenities: {host_amenities}
- Their composite ranking score: {my_score} (beats {percentile}% of competitors in {host_neighbourhood})
- Neighbourhood median price: €{nb_median:.0f}
- Competing listings in area: {nb_listings}

Generate 5 highly specific, actionable recommendations for this host:
1. Price positioning: are they over or underpriced vs the €{nb_median:.0f} median?
2. How their {host_reviews} reviews compare to top performers
3. Which of their amenities ({host_amenities}) are strongest differentiators
4. What they should change THIS WEEK to improve their ranking
5. One optimised listing description opening sentence that highlights their best feature

Be direct, specific, and use the numbers provided.
"""
            llm = ChatOpenAI(model="gpt-4o-mini")
            response = llm.invoke([HumanMessage(content=personal_prompt)])
            personal_insights = response.content

        col_score, col_insights = st.columns([1, 2])
        with col_score:
            score_color = "#00D4FF" if percentile >= 50 else "#FFD700" if percentile >= 25 else "#FF6B6B"
            st.markdown(f"""
            <div style="background: #132140; border: 1px solid rgba(0,212,255,0.2); border-radius: 12px; padding: 1.5rem; text-align: center;">
                <p style="color: #8899AA; font-size: 0.7rem; font-family: Space Mono, monospace; letter-spacing: 0.1em; margin: 0 0 0.5rem;">YOUR SCORE</p>
                <p style="color: {score_color}; font-size: 3rem; font-weight: 700; margin: 0; line-height: 1;">{my_score}</p>
                <p style="color: #8899AA; font-size: 0.85rem; margin: 0.5rem 0 1.5rem;">beats {percentile}% of {host_neighbourhood} listings</p>
                <div style="border-top: 1px solid rgba(0,212,255,0.1); padding-top: 1rem;">
                    <p style="color: #8899AA; font-size: 0.7rem; font-family: Space Mono, monospace; margin: 0 0 0.3rem;">SCORE BREAKDOWN</p>
                    <p style="color: #C8D8E8; font-size: 0.8rem; margin: 0.2rem 0;">Price &nbsp;&nbsp; {price_score:.3f} x 40%</p>
                    <p style="color: #C8D8E8; font-size: 0.8rem; margin: 0.2rem 0;">Reviews &nbsp; {review_score:.3f} x 40%</p>
                    <p style="color: #C8D8E8; font-size: 0.8rem; margin: 0.2rem 0;">Demand &nbsp; {demand_score:.3f} x 20%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_insights:
            st.markdown(f'<div class="insights-box">{personal_insights}</div>', unsafe_allow_html=True)
        st.toast("✓ Your listing has been analysed!", icon="🏠")

    # Footer
    st.markdown("""
    <div class="footer">
        BerlinHostAIQ &nbsp;·&nbsp; Antonio Ciraci &nbsp;·&nbsp; AI Bootcamp Final Project &nbsp;·&nbsp; March 2026
        &nbsp;·&nbsp; Data: Inside Airbnb (public)
        &nbsp;·&nbsp; Model: GPT-4o-mini
        &nbsp;·&nbsp; Tracing: LangSmith
        &nbsp;·&nbsp; <a href="https://github.com/ciraci2-netizen/final-project-antonio-ciraci" style="color: #00D4FF; text-decoration: none;">GitHub &#8599;</a>
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

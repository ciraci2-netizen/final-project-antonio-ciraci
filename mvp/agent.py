from dotenv import load_dotenv
import os
import pandas as pd
import warnings
from datetime import datetime

load_dotenv()

try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
except Exception:
    pass

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "short-term-rental-analysis"
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import Client


def load_data(path=None):
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "data", "raw", "listings.csv")
    try:
        df = pd.read_csv(path)
        df["price"] = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True).pipe(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["price", "number_of_reviews", "neighbourhood_group"])
        df["number_of_reviews"] = df["number_of_reviews"].astype(int)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {path}.")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def compute_ranking(df, neighbourhood_filter=None):
    working = df.copy()
    if neighbourhood_filter and neighbourhood_filter != "All":
        working = working[working["neighbourhood_group"] == neighbourhood_filter]
    if working.empty:
        return working
    neighbourhood_median = working.groupby("neighbourhood_group")["price"].transform("median")
    working["price_score"] = 1 - (working["price"] / (neighbourhood_median * 2)).clip(0, 1)

    max_reviews = working["number_of_reviews"].max()
    avg_monthly = working["reviews_per_month"].mean()

    quality_score = (working["reviews_per_month"] / max(avg_monthly, 0.1)).clip(0, 1)
    quantity_score = working["number_of_reviews"] / max_reviews if max_reviews > 0 else 0

    working["review_score"] = (quality_score * 0.6 + quantity_score * 0.4)

    neighbourhood_counts = working["neighbourhood_group"].map(working["neighbourhood_group"].value_counts())
    import numpy as np
    log_counts = np.log1p(neighbourhood_counts)
    working["demand_score"] = log_counts / log_counts.max()
    working["ranking_score"] = (working["price_score"] * 0.4 + working["review_score"] * 0.4 + working["demand_score"] * 0.2).round(3)
    return working.sort_values("ranking_score", ascending=False)


def create_langsmith_dataset(sample):
    try:
        client = Client()
        dataset_name = f"airbnb_listings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset = client.create_dataset(dataset_name=dataset_name, description="Berlin Airbnb listings — ranked sample")
        for _, row in sample.iterrows():
            client.create_example(
                dataset_id=dataset.id,
                inputs={"name": str(row.get("name", "N/A")), "price": float(row["price"]),
                        "number_of_reviews": int(row["number_of_reviews"]),
                        "neighbourhood_group": str(row["neighbourhood_group"]),
                        "room_type": str(row.get("room_type", "N/A")),
                        "ranking_score": float(row.get("ranking_score", 0))},
                outputs={"insight_type": "competitive_ranking"}
            )
        return dataset.id
    except Exception as e:
        print(f"LangSmith error: {e}")
        return None


def analyze_listings(sample_df, neighbourhood_filter="All"):
    top = sample_df.head(10)
    cols = ["name", "price", "number_of_reviews", "neighbourhood_group", "room_type", "ranking_score"]
    available_cols = [c for c in cols if c in top.columns]
    data_text = top[available_cols].to_string(index=False)
    neighbourhood_context = f"Focus area: {neighbourhood_filter} neighbourhood." if neighbourhood_filter != "All" else "Analysis covers all Berlin neighbourhoods."
    prompt = f"""You are a competitive intelligence analyst for the Berlin short-term rental market.
{neighbourhood_context}
Top 10 ranked listings:
{data_text}
Generate exactly 5 actionable insights for an independent host. Focus on: pricing vs median, review demand signals, key differentiators, risks/opportunities, one concrete action this week. Be specific and data-driven.
IMPORTANT: Start directly with "1." — no preamble, no meta-commentary, no intro sentence."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"AI analysis error: {e}"


def recommend_optimal_price(host_price, neighbourhood_df, host_room_type=None):
    """Calculate optimal price based on market data — FIX 6: filter by room type"""
    working = neighbourhood_df.copy()
    # FIX 6: filter to same room type for a fair comparison
    if host_room_type and host_room_type != "All" and "room_type" in working.columns:
        filtered = working[working["room_type"] == host_room_type]
        if not filtered.empty:
            working = filtered

    if working.empty:
        return None, None, None

    median_price = working["price"].median()
    percentile_75 = working["price"].quantile(0.75)

    recommended_low = max(median_price * 0.85, working["price"].min())
    recommended_high = min(percentile_75 * 1.05, working["price"].max())
    recommended_mid = (recommended_low + recommended_high) / 2

    return round(recommended_mid, 0), round(recommended_low, 0), round(recommended_high, 0)


def format_insights_compact(insights_text):
    import re
    points = re.split(r'\n\s*\d+\.\s+', insights_text)
    if points and not re.match(r'^[A-Z]', points[0]):
        header = points[0]
        points = points[1:]
    else:
        header = None
    return points[:5], header


def display_insights_expander(insights_text, title="🤖 AI Recommendations"):
    import streamlit as st
    points, header = format_insights_compact(insights_text)
    with st.expander(title, expanded=True):
        if header:
            st.markdown(f"*{header}*")
        for idx, point in enumerate(points, 1):
            point_clean = point.strip()
            if point_clean:
                lines = point_clean.split('\n')
                point_title = lines[0][:60].strip()
                point_detail = '\n'.join(lines)
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    st.markdown(f"**{idx}.**")
                with col2:
                    st.markdown(f"_{point_title}_")
                    st.caption(point_detail)


def get_top_competitors(host_price, host_neighbourhood, host_room_type, df, comp_df_ranked):
    """Find 3 most similar direct competitors"""
    if comp_df_ranked is not None and not comp_df_ranked.empty:
        comp_df = comp_df_ranked.copy()
    else:
        if host_neighbourhood == "All":
            comp_df = df.copy()
        else:
            comp_df = df[df["neighbourhood_group"] == host_neighbourhood].copy()

    # FIX 3 prep: always filter by room type for fair comparison
    if host_room_type and host_room_type != "All" and "room_type" in comp_df.columns:
        rt_filtered = comp_df[comp_df["room_type"] == host_room_type]
        if not rt_filtered.empty:
            comp_df = rt_filtered

    price_range = host_price * 0.2
    comp_df = comp_df[
        (comp_df["price"] >= host_price - price_range) &
        (comp_df["price"] <= host_price + price_range)
    ].copy()

    if comp_df.empty:
        return pd.DataFrame()

    if "ranking_score" in comp_df.columns:
        comp_df["similarity"] = comp_df["ranking_score"] - (comp_df["price"] - host_price).abs() / 100
    else:
        comp_df["similarity"] = -(comp_df["price"] - host_price).abs()

    cols = ["name", "neighbourhood_group", "room_type", "price", "number_of_reviews"]
    if "ranking_score" in comp_df.columns:
        cols.append("ranking_score")
    available_cols = [c for c in cols if c in comp_df.columns]

    return comp_df.nlargest(3, "similarity")[available_cols].reset_index(drop=True)


def create_price_strategy_guide(df):
    import plotly.graph_objects as go
    zones = {
        "Budget 🟢": {"range": [df["price"].min(), 50], "color": "#90EE90", "description": "Max Occupancy"},
        "Mid-Range 🟡": {"range": [50, 80], "color": "#FFD700", "description": "Sweet Spot"},
        "Premium 🔵": {"range": [80, df["price"].max()], "color": "#00D4FF", "description": "High Margin"}
    }
    zone_stats = []
    for zone_name, zone_info in zones.items():
        min_price, max_price = zone_info["range"]
        zone_data = df[(df["price"] >= min_price) & (df["price"] < max_price)]
        if not zone_data.empty:
            zone_stats.append({
                "Zone": zone_name, "Count": len(zone_data),
                "Pct": round((len(zone_data) / len(df)) * 100, 1),
                "Avg Score": round(zone_data["ranking_score"].mean(), 3),
                "Avg Reviews": round(zone_data["number_of_reviews"].mean(), 0),
                "Price Range": f"€{min_price:.0f}-€{max_price:.0f}",
                "Color": zone_info["color"], "Description": zone_info["description"]
            })
    zone_df = pd.DataFrame(zone_stats)
    fig = go.Figure()
    for idx, row in zone_df.iterrows():
        fig.add_trace(go.Bar(
            y=[row["Zone"]], x=[row["Count"]], orientation='h',
            marker=dict(color=row["Color"]),
            text=f"{row['Pct']}% | {int(row['Count'])} listings<br>Avg Score: {row['Avg Score']}<br>Avg Reviews: {int(row['Avg Reviews'])}<br>{row['Price Range']}",
            textposition="auto", textfont=dict(color="#0A1628", size=11, family="Space Mono"),
            hovertemplate=f"<b>{row['Zone']}</b><br>Listings: {int(row['Count'])} ({row['Pct']}%)<br>Avg Score: {row['Avg Score']}<br>Avg Reviews: {int(row['Avg Reviews'])}<extra></extra>",
            name=row["Zone"], showlegend=False
        ))
    fig.update_layout(
        paper_bgcolor="#0A1628", plot_bgcolor="#0A1628",
        font=dict(color="#C8D8E8", family="Space Mono"), height=250,
        margin=dict(l=100, r=20, t=20, b=20),
        xaxis=dict(title="Number of Listings", gridcolor="rgba(0,212,255,0.08)", color="#8899AA"),
        yaxis=dict(color="#C8D8E8", tickfont=dict(size=12, family="Space Mono")),
    )
    return fig, zone_df


def create_growth_roadmap(host_reviews, host_price, host_neighbourhood, comp_df):
    if host_reviews >= 100:
        current_tier = "Bestseller 🌟"; target_tier = "Elite 👑"; target_reviews = 150
    elif host_reviews >= 50:
        current_tier = "Established ⭐"; target_tier = "Bestseller 🌟"; target_reviews = 100
    elif host_reviews >= 20:
        current_tier = "Rising Star 📈"; target_tier = "Established ⭐"; target_reviews = 50
    else:
        current_tier = "New Host 🆕"; target_tier = "Rising Star 📈"; target_reviews = 20
    reviews_needed = max(0, target_reviews - host_reviews)
    monthly_rate = comp_df[comp_df['number_of_reviews'] > 0]['reviews_per_month'].mean()
    months_needed = max(1, reviews_needed / max(monthly_rate, 0.5))
    milestone_1 = host_reviews + (monthly_rate * max(1, months_needed / 3))
    milestone_2 = host_reviews + (monthly_rate * max(1, months_needed * 2/3))
    milestone_3 = target_reviews
    return {
        "current_tier": current_tier, "target_tier": target_tier,
        "target_reviews": target_reviews, "reviews_needed": reviews_needed,
        "monthly_rate": monthly_rate, "months_needed": months_needed,
        "milestone_1": milestone_1, "milestone_2": milestone_2, "milestone_3": milestone_3
    }


def calculate_review_quality(host_reviews, host_reviews_per_month, comp_df):
    avg_monthly = comp_df["reviews_per_month"].mean()
    max_reviews = comp_df["number_of_reviews"].max()

    # Velocity: how does host's booking rate compare to market average?
    velocity_score = min(1.0, (host_reviews_per_month / max(avg_monthly, 0.1)))

    # Consistency: based on host's actual reviews/month
    if host_reviews_per_month >= avg_monthly:
        consistency_score = 0.9
    elif host_reviews_per_month >= avg_monthly * 0.5:
        consistency_score = 0.7
    elif host_reviews_per_month > 0:
        consistency_score = 0.5
    else:
        consistency_score = 0.3

    # Recency: proxy — higher reviews/month = more recent activity
    recency_score = min(1.0, (host_reviews_per_month / max(avg_monthly, 0.1)))

    quality_score = (velocity_score * 0.5 + consistency_score * 0.35 + recency_score * 0.15)
    return {
        "velocity_score": velocity_score, "consistency_score": consistency_score,
        "recency_score": recency_score, "quality_score": quality_score,
        "status": "🟢 Excellent" if quality_score >= 0.8 else "🟡 Good" if quality_score >= 0.6 else "🔴 Needs Work"
    }


def calculate_kpi_metrics(listing_row):
    from datetime import datetime
    price = listing_row["price"]
    availability = listing_row["availability_365"]
    reviews_ltm = listing_row["number_of_reviews_ltm"]
    reviews_per_month = listing_row["reviews_per_month"]
    last_review = listing_row["last_review"]
    occupancy_rate = (365 - availability) / 365 * 100
    adr = price
    revpar = (occupancy_rate / 100) * adr
    days_since_last_review = "N/A"
    if pd.notna(last_review) and str(last_review) != "nan":
        try:
            last_review_date = pd.to_datetime(last_review)
            days_since = (datetime.now() - last_review_date).days
            days_since_last_review = days_since
        except:
            days_since_last_review = "N/A"
    return {
        "occupancy_rate": occupancy_rate, "adr": adr, "revpar": revpar,
        "reviews_ltm": reviews_ltm, "days_since_last_review": days_since_last_review,
        "reviews_per_month": reviews_per_month
    }


def calculate_revenue_impact(host_price, optimal_price, current_reviews_per_month, market_avg_reviews_per_month, occupancy_rate):
    price_diff = optimal_price - host_price
    monthly_bookings = max(current_reviews_per_month, 1)
    monthly_revenue_loss = price_diff * monthly_bookings * 30
    annual_revenue_impact = monthly_revenue_loss * 12
    activity_ratio = market_avg_reviews_per_month / max(current_reviews_per_month, 0.1) if current_reviews_per_month > 0 else 2.0
    activity_ratio = min(activity_ratio, 3.0)
    potential_monthly_bookings = monthly_bookings * activity_ratio
    revenue_gain_from_activity = (potential_monthly_bookings - monthly_bookings) * optimal_price * 30
    total_potential = monthly_revenue_loss + (revenue_gain_from_activity / 12)
    return {
        "monthly_loss_from_price": monthly_revenue_loss,
        "annual_loss_from_price": annual_revenue_impact,
        "monthly_gain_from_activity": revenue_gain_from_activity / 12,
        "monthly_total_potential": total_potential,
        "annual_total_potential": total_potential * 12,
        "price_diff": price_diff,
        "loss_percentage": (abs(monthly_revenue_loss) / (host_price * monthly_bookings * 30) * 100) if host_price > 0 else 0
    }


def create_report_breakdown_saas(revenue_impact, recommended_price, current_price, host_reviews, positioning_rec, market_stats):
    sections = []
    price_gap = recommended_price - current_price
    price_gap_pct = (price_gap / current_price * 100) if current_price > 0 else 0
    pricing_section = f"""
    <div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:1.5rem; margin-bottom:1.5rem;">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
            <div>
                <p style="color:#8899AA; font-size:0.7rem; text-transform:uppercase; margin:0 0 0.5rem; letter-spacing:0.1em;"><b>💰 Pricing Gap</b></p>
                <p style="color:#FF6B6B; font-size:2.2rem; font-weight:700; margin:0;">€{price_gap:.0f}<span style="font-size:1rem; color:#8899AA;">/night</span></p>
                <p style="color:#C8D8E8; font-size:0.85rem; margin:0.5rem 0;">({price_gap_pct:+.0f}% vs current)</p>
            </div>
            <div>
                <p style="color:#8899AA; font-size:0.7rem; text-transform:uppercase; margin:0 0 0.5rem; letter-spacing:0.1em;"><b>📈 Monthly Impact</b></p>
                <p style="color:#90EE90; font-size:2.2rem; font-weight:700; margin:0;">€{revenue_impact['monthly_total_potential']:.0f}</p>
                <p style="color:#C8D8E8; font-size:0.85rem; margin:0.5rem 0;">extra per month</p>
            </div>
        </div>
        <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(0,212,255,0.1);">
            <p style="color:#FFD700; font-size:0.85rem; margin:0;"><b>🎯 Action:</b> Start with +€{min(price_gap * 0.5, 20):.0f}/night. Track bookings for 2 weeks.</p>
        </div>
    </div>
    """
    sections.append(pricing_section)
    velocity_target = market_stats['median_activity'] if 'median_activity' in market_stats else 0.7
    velocity_section = f"""
    <div style="background:#132140; border:1px solid rgba(255,215,0,0.2); border-radius:8px; padding:1.5rem; margin-bottom:1.5rem;">
        <p style="color:#FFD700; font-size:0.7rem; text-transform:uppercase; margin:0 0 1rem; letter-spacing:0.1em;"><b>⚡ Review Velocity</b></p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
            <div>
                <p style="color:#8899AA; font-size:0.75rem; margin:0 0 0.3rem;">Your Reviews/Month</p>
                <p style="color:#FFD700; font-size:1.8rem; font-weight:700; margin:0;">0.{int(host_reviews % 10 or 5)}</p>
            </div>
            <div>
                <p style="color:#8899AA; font-size:0.75rem; margin:0 0 0.3rem;">Market Target</p>
                <p style="color:#90EE90; font-size:1.8rem; font-weight:700; margin:0;">{velocity_target:.1f}</p>
                <p style="color:#C8D8E8; font-size:0.65rem; margin:0.3rem 0;">reviews/month</p>
            </div>
        </div>
        <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(255,215,0,0.1);">
            <p style="color:#FFD700; font-size:0.85rem; margin:0;"><b>🔄 Action:</b> Improve response time to &lt;1 hour. Add availability for weekends.</p>
        </div>
    </div>
    """
    sections.append(velocity_section)
    positioning_section = f"""
    <div style="background:#132140; border:1px solid rgba(144,238,144,0.2); border-radius:8px; padding:1.5rem; margin-bottom:1.5rem;">
        <p style="color:#90EE90; font-size:0.7rem; text-transform:uppercase; margin:0 0 1rem; letter-spacing:0.1em;"><b>🎯 Strategic Positioning</b></p>
        <div style="background:rgba(144,238,144,0.05); border-left:3px solid #90EE90; padding:1rem; border-radius:0 8px 8px 0;">
            <p style="color:#90EE90; font-size:1rem; font-weight:700; margin:0 0 0.5rem;">{positioning_rec}</p>
            <p style="color:#C8D8E8; font-size:0.85rem; margin:0;">Focus on this segment to differentiate from competitors.</p>
        </div>
        <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid rgba(144,238,144,0.1);">
            <p style="color:#90EE90; font-size:0.85rem; margin:0;"><b>💡 Action:</b> Update listing title &amp; description to emphasize this positioning.</p>
        </div>
    </div>
    """
    sections.append(positioning_section)
    quick_wins = f"""
    <div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:1.5rem;">
        <p style="color:#00D4FF; font-size:0.7rem; text-transform:uppercase; margin:0 0 1rem; letter-spacing:0.1em;"><b>⚡ Quick Wins (This Week)</b></p>
        <div style="display:grid; gap:0.8rem;">
            <div style="background:rgba(0,212,255,0.05); padding:0.8rem; border-radius:6px; border-left:2px solid #00D4FF;">
                <p style="color:#00D4FF; font-weight:700; font-size:0.85rem; margin:0;"> 1. Update pricing</p>
                <p style="color:#C8D8E8; font-size:0.75rem; margin:0;">Adjust nightly rate to €{recommended_price:.0f}</p>
            </div>
            <div style="background:rgba(0,212,255,0.05); padding:0.8rem; border-radius:6px; border-left:2px solid #00D4FF;">
                <p style="color:#00D4FF; font-weight:700; font-size:0.85rem; margin:0;"> 2. Adjust availability</p>
                <p style="color:#C8D8E8; font-size:0.75rem; margin:0;">Open weekend bookings for higher demand</p>
            </div>
            <div style="background:rgba(0,212,255,0.05); padding:0.8rem; border-radius:6px; border-left:2px solid #00D4FF;">
                <p style="color:#00D4FF; font-weight:700; font-size:0.85rem; margin:0;"> 3. Refresh photos</p>
                <p style="color:#C8D8E8; font-size:0.75rem; margin:0;">Add natural lighting &amp; lifestyle images</p>
            </div>
        </div>
    </div>
    """
    sections.append(quick_wins)
    return "\n".join(sections)


def create_competitive_comparison(your_listing, competitors_df):
    comparison_data = []
    your_data = {
        "Listing": "🏆 YOUR LISTING",
        "Price (€/night)": f"€{your_listing['price']:.0f}",
        "Total Reviews": f"{your_listing['number_of_reviews']}",
        "Reviews/Month": f"{your_listing['reviews_per_month']:.2f}",
        "Room Type": your_listing['room_type'],
        "Availability": f"{365 - your_listing['availability_365']} days"
    }
    comparison_data.append(your_data)
    for idx, (_, comp) in enumerate(competitors_df.head(3).iterrows(), 1):
        comp_data = {
            "Listing": f"#{idx} Competitor",
            "Price (€/night)": f"€{comp['price']:.0f}",
            "Total Reviews": f"{comp['number_of_reviews']}",
            "Reviews/Month": f"{comp['reviews_per_month']:.2f}",
            "Room Type": comp['room_type'],
            "Availability": f"{365 - comp['availability_365']} days"
        }
        comparison_data.append(comp_data)
    return pd.DataFrame(comparison_data)


def analyze_positioning_opportunity(listing_df, neighbourhood, room_type):
    same_market = listing_df[
        (listing_df['neighbourhood'] == neighbourhood) &
        (listing_df['room_type'] == room_type) &
        (listing_df['price'].notna()) &
        (listing_df['reviews_per_month'].notna())
    ]
    if len(same_market) == 0:
        return None
    price_q25 = same_market['price'].quantile(0.25)
    price_q75 = same_market['price'].quantile(0.75)
    activity_q50 = same_market['reviews_per_month'].median()
    positionings = {
        "🏡 Family-friendly": {"price_range": (price_q25, price_q75), "ideal_activity": activity_q50, "description": "Spacious, welcoming for families with children"},
        "💼 Smart Working": {"price_range": (price_q75 * 0.8, price_q75 * 1.2), "ideal_activity": activity_q50 * 1.1, "description": "High-speed WiFi, dedicated workspace, quiet"},
        "✨ Luxury/Design": {"price_range": (price_q75 * 1.1, price_q75 * 2), "ideal_activity": activity_q50 * 0.8, "description": "Premium design, exclusive experience, premium amenities"},
        "💰 Budget Efficient": {"price_range": (price_q25 * 0.7, price_q25 * 1.1), "ideal_activity": activity_q50 * 1.3, "description": "Affordable, high occupancy, good value"},
        "🌍 Local Experience": {"price_range": (price_q25, price_q75 * 1.5), "ideal_activity": activity_q50, "description": "Authentic Berlin, local knowledge, neighborhood integration"}
    }
    return {
        "market_stats": {
            "avg_price": same_market['price'].mean(),
            "median_price": same_market['price'].median(),
            "avg_activity": same_market['reviews_per_month'].mean(),
            "median_activity": same_market['reviews_per_month'].median(),
            "total_listings": len(same_market),
            "avg_reviews": same_market['number_of_reviews'].mean()
        },
        "positionings": positionings
    }


def get_positioning_recommendation(listing_price, listing_activity, market_stats):
    price_ratio = listing_price / market_stats['median_price']
    activity_ratio = listing_activity / market_stats['median_activity'] if market_stats['median_activity'] > 0 else 0
    if price_ratio < 0.85:
        if activity_ratio > 1.2:
            return "💰 Budget Efficient", "Strong volume positioning - high demand at competitive price"
        else:
            return "💰 Budget Efficient", "Competitive pricing - increase visibility to boost bookings"
    elif price_ratio <= 1.15:
        if activity_ratio > 1.0:
            return "🏡 Family-friendly", "Mid-market with strong activity - trusted positioning"
        else:
            return "🌍 Local Experience", "Mid-market opportunity - emphasize unique local experience"
    else:
        if activity_ratio >= 0.8:
            return "✨ Luxury/Design", "Premium positioning with consistent demand - premium amenities matter"
        else:
            return "✨ Luxury/Design", "Premium price - enhance listing with luxury positioning to justify rate"


def calculate_revenue_scenarios(host_price, comp_df, neighbourhood_median):
    price_ratio = host_price / neighbourhood_median
    if price_ratio < 0.8:
        occupancy_base = 0.65
    elif price_ratio < 1.0:
        occupancy_base = 0.55
    elif price_ratio < 1.2:
        occupancy_base = 0.45
    else:
        occupancy_base = 0.35
    scenarios = []
    prices = [host_price - 20, host_price - 10, host_price, host_price + 10, host_price + 20]
    for price in prices:
        price = max(10, price)
        price_diff_pct = (price - host_price) / host_price if host_price > 0 else 0
        occupancy_adj = occupancy_base * (1 - price_diff_pct * 0.3)
        occupancy = max(0.1, min(0.95, occupancy_adj))
        monthly_revenue = price * 30 * occupancy
        annual_revenue = monthly_revenue * 12
        cheaper_listings = len(comp_df[comp_df['price'] < price])
        rank_position = cheaper_listings
        scenarios.append({
            "Price": f"€{int(price)}",
            "Monthly": f"€{int(monthly_revenue)}",
            "Annual": f"€{int(annual_revenue)}",
            "Occupancy": f"{occupancy * 100:.0f}%",
            "Position": f"#{rank_position}",
            "is_current": abs(price - host_price) < 1
        })
    return scenarios


def analyze_review_intelligence(df):
    df_analysis = df.dropna(subset=['number_of_reviews', 'reviews_per_month']).copy()
    if df_analysis.empty:
        return None, None, None, None
    df_analysis['review_velocity'] = df_analysis['number_of_reviews'] / (df_analysis['availability_365'] + 1) * 365
    def categorize_tier(reviews):
        if reviews >= 100: return "Bestseller 🌟"
        elif reviews >= 50: return "Established ⭐"
        elif reviews >= 20: return "Rising Star 📈"
        else: return "New Host 🆕"
    df_analysis['host_tier'] = df_analysis['number_of_reviews'].apply(categorize_tier)
    df_analysis['review_conversion'] = (df_analysis['number_of_reviews'] / (366 - df_analysis['availability_365'] + 1)).clip(0, 1)
    tier_stats = df_analysis.groupby('host_tier').agg({
        'number_of_reviews': ['mean', 'count'], 'price': 'mean',
        'ranking_score': 'mean', 'reviews_per_month': 'mean', 'review_conversion': 'mean'
    }).round(2)
    top_perf = df_analysis[df_analysis['number_of_reviews'] >= df_analysis['number_of_reviews'].quantile(0.75)]
    return tier_stats, df_analysis, top_perf, df_analysis['host_tier'].value_counts()


LANG = {
    "EN": {
        "hero_title": "BerlinHostAIQ", "hero_subtitle": "💰 Discover how much you're losing on your Airbnb — and how to earn more",
        "hero_analysis": "✅ Analysis in 30 seconds", "hero_desc1": "We compare your listing vs 9,264 Berlin properties",
        "hero_desc2": "Discover: optimal price, competitors, growth strategies",
        "filters": "Filters", "neighbourhood": "Neighbourhood", "room_type": "Room type",
        "top_n": "Top listings to analyse", "rankings": "Top Ranked Listings",
        "caption": "Score = Price competitiveness (40%) + Review velocity (40%) + Neighbourhood demand (20%)",
        "map_title": "Berlin Neighbourhoods — Price & Demand", "chart_title": "Median Price by Neighbourhood",
        "insights_title": "AI Competitive Insights", "generate": "Generate Insights →",
        "analyse_title": "Analyse My Listing", "how_it_works": "HOW IT WORKS",
        "how_text": "Enter your listing details — Discover instantly how much you're losing and how to earn more money.",
        "price_label": "Your nightly price (€)", "nb_label": "Your neighbourhood",
        "reviews_label": "Number of reviews",
        "amenities_label": "Key amenities (e.g. balcony, garden, parking, wifi)",
        "submit": "💰 Discover My Potential →", "your_score": "YOUR SCORE", "beats": "beats",
        "breakdown": "SCORE BREAKDOWN", "total": "Total Listings", "median": "Median Price",
        "avg": "Avg Reviews", "top": "Top Score",
        "spinner_market": "Analysing market data with GPT-4o-mini...",
        "spinner_listing": "Ranking your listing vs competitors...",
        "toast_market": "✓ Analysis complete!", "toast_listing": "✓ Your listing has been analysed!", "night": "/night",
    },
    "IT": {
        "hero_title": "BerlinHostAIQ", "hero_subtitle": "💰 Scopri quanto stai perdendo sul tuo Airbnb — e come guadagnare di più",
        "hero_analysis": "✅ Analisi in 30 secondi", "hero_desc1": "Confrontiamo il tuo annuncio vs 9,264 listing di Berlino",
        "hero_desc2": "Scopri: prezzo ottimale, competitors, strategie di growth",
        "filters": "Filtri", "neighbourhood": "Quartiere", "room_type": "Tipo di stanza",
        "top_n": "Top listing da analizzare", "rankings": "Top Listing per Ranking",
        "caption": "Score = Competitivita prezzo (40%) + Velocita recensioni (40%) + Domanda quartiere (20%)",
        "map_title": "Quartieri di Berlino — Prezzi & Domanda", "chart_title": "Prezzo Mediano per Quartiere",
        "insights_title": "AI Competitive Insights", "generate": "Genera Insights →",
        "analyse_title": "Analizza il mio Listing", "how_it_works": "COME FUNZIONA",
        "how_text": "Inserisci prezzo, quartiere e recensioni — scoprirai subito quanto stai perdendo e come recuperare i soldi persi.",
        "price_label": "Prezzo a notte (€)", "nb_label": "Il tuo quartiere",
        "reviews_label": "Numero di recensioni",
        "amenities_label": "Amenities principali (es. balcone, giardino, parcheggio, wifi)",
        "submit": "💰 Scopri il mio potenziale →", "your_score": "IL TUO SCORE", "beats": "supera",
        "breakdown": "DETTAGLIO SCORE", "total": "Listing Totali", "median": "Prezzo Mediano",
        "avg": "Recensioni Medie", "top": "Top Score",
        "spinner_market": "Analisi dati di mercato con GPT-4o-mini...",
        "spinner_listing": "Posizionamento del tuo listing vs competitor...",
        "toast_market": "✓ Analisi completata!", "toast_listing": "✓ Il tuo listing e stato analizzato!", "night": "/notte",
    },
    "DE": {
        "hero_title": "BerlinHostAIQ", "hero_subtitle": "💰 Entdecken Sie, wie viel Sie mit Ihrem Airbnb verlieren — und wie Sie mehr verdienen",
        "hero_analysis": "✅ Analyse in 30 Sekunden", "hero_desc1": "Wir vergleichen Ihr Inserat mit 9.264 Berliner Objekten",
        "hero_desc2": "Entdecken Sie: optimaler Preis, Konkurrenten, Wachstumsstrategien",
        "filters": "Filter", "neighbourhood": "Bezirk", "room_type": "Zimmertyp",
        "top_n": "Top-Inserate analysieren", "rankings": "Top-Inserate nach Ranking",
        "caption": "Score = Preiswettbewerb (40%) + Bewertungsgeschwindigkeit (40%) + Bezirksnachfrage (20%)",
        "map_title": "Berliner Bezirke — Preise & Nachfrage", "chart_title": "Medianpreis nach Bezirk",
        "insights_title": "KI-Wettbewerbsanalyse", "generate": "Insights generieren →",
        "analyse_title": "Mein Inserat analysieren", "how_it_works": "SO FUNKTIONIERT ES",
        "how_text": "Geben Sie Ihren Preis, Bezirk und Bewertungen ein — entdecken Sie sofort, wie viel Sie verlieren und wie Sie mehr verdienen.",
        "price_label": "Ihr Nachtpreis (€)", "nb_label": "Ihr Bezirk",
        "reviews_label": "Anzahl Bewertungen",
        "amenities_label": "Hauptausstattung (z.B. Balkon, Garten, Parkplatz, WLAN)",
        "submit": "💰 Mein Potenzial entdecken →", "your_score": "IHR SCORE", "beats": "besser als",
        "breakdown": "SCORE-AUFSCHLUSSELUNG", "total": "Inserate gesamt", "median": "Medianpreis",
        "avg": "Bewertungen", "top": "Top Score",
        "spinner_market": "Marktdaten werden analysiert...", "spinner_listing": "Ihr Inserat wird verglichen...",
        "toast_market": "✓ Analyse abgeschlossen!", "toast_listing": "✓ Ihr Inserat wurde analysiert!", "night": "/Nacht",
    },
}


def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="BerlinHostAIQ — Competitive Intelligence for Airbnb Hosts",
        page_icon="🏠", layout="wide", initial_sidebar_state="expanded"
    )

    if "language" not in st.session_state:
        st.session_state.language = "EN"

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
        :root { --bg-primary:#0A1628; --bg-secondary:#0F1F38; --bg-card:#132140; --cyan:#00D4FF; --gold:#FFD700; --white:#FFFFFF; --muted:#8899AA; --border:rgba(0,212,255,0.15); }
        .stApp { background-color:var(--bg-primary); font-family:'DM Sans',sans-serif; }
        [data-testid='stDataFrame'] { border:1px solid rgba(0,212,255,0.15) !important; border-radius:8px; overflow:hidden; }
        [data-testid='stDataFrame'] th { background:#0F1F38 !important; color:#00D4FF !important; font-family:'Space Mono',monospace !important; font-size:0.68rem !important; letter-spacing:0.05em !important; text-transform:uppercase !important; }
        [data-testid='stDataFrame'] td { border-bottom:1px solid rgba(0,212,255,0.05) !important; }
        .block-container { padding-top:1rem !important; }
        header[data-testid="stHeader"] { background-color:var(--bg-primary) !important; }
        [data-testid="stSidebar"] { background-color:var(--bg-secondary) !important; border-right:1px solid var(--border); }
        [data-testid="stSidebar"] * { color:var(--white) !important; }
        .hero-header { padding:1.5rem 0; border-bottom:1px solid var(--border); margin-bottom:2rem; }
        .hero-title { font-family:'Space Mono',monospace; font-size:2.4rem; font-weight:700; color:var(--white); letter-spacing:-0.02em; margin:0; line-height:1.1; }
        .hero-title span { color:var(--cyan); }
        .hero-subtitle { font-size:0.95rem; color:var(--muted); margin-top:0.5rem; }
        .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2rem; }
        .kpi-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:1.2rem 1.5rem; position:relative; overflow:hidden; }
        .kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--cyan),transparent); }
        .kpi-label { font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; font-family:'Space Mono',monospace; margin-bottom:0.4rem; }
        .kpi-value { font-size:1.3rem; font-weight:600; color:var(--white); line-height:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .kpi-value.cyan { color:var(--cyan); }
        .kpi-value.gold { color:var(--gold); }
        .section-header { font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; color:var(--cyan); letter-spacing:0.12em; text-transform:uppercase; margin:2rem 0 1rem; display:flex; align-items:center; gap:10px; }
        .section-header::after { content:''; flex:1; height:1px; background:var(--border); }
        .insights-box { background:var(--bg-card); border:1px solid var(--border); border-left:3px solid var(--cyan); border-radius:0 12px 12px 0; padding:1rem 1.5rem; margin-top:1rem; color:#C8D8E8; line-height:1.4; font-size:0.85rem; white-space:pre-wrap; }
        .footer { margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border); font-size:0.75rem; color:var(--muted); font-family:'Space Mono',monospace; }
        p, li, span, label { color:#C8D8E8 !important; }
        h1, h2, h3 { color:var(--white) !important; }
        .stButton > button { background:linear-gradient(135deg,var(--cyan),#0099CC) !important; color:#0A1628 !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; font-size:0.85rem !important; border:none !important; border-radius:8px !important; padding:0.7rem 2rem !important; }
        div[data-baseweb="select"] > div { background-color:#132140 !important; border-color:rgba(0,212,255,0.3) !important; color:#E0EEFF !important; }
        div[data-baseweb="select"] span { color:#E0EEFF !important; }
        div[data-baseweb="select"] svg { fill:#00D4FF !important; }
        div[data-baseweb="popover"] div[role="listbox"] { background-color:#0F1F38 !important; border:1px solid rgba(0,212,255,0.2) !important; }
        div[data-baseweb="popover"] li { background-color:#0F1F38 !important; color:#E0EEFF !important; }
        div[data-baseweb="popover"] li:hover { background-color:rgba(0,212,255,0.15) !important; color:#00D4FF !important; }
        div[data-baseweb="popover"] li[aria-selected="true"] { background-color:rgba(0,212,255,0.2) !important; color:#00D4FF !important; }
    </style>
    """, unsafe_allow_html=True)

    try:
        df = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        new_lang = st.radio("🌐 Language", ["EN", "IT", "DE"], horizontal=True, index=["EN", "IT", "DE"].index(st.session_state.language))
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()
        T = LANG[st.session_state.language]

    T_hero = LANG[st.session_state.language]
    st.markdown(f"""
    <div class="hero-header">
        <p class="hero-title">Berlin<span>Host</span>AIQ</p>
        <p class="hero-subtitle">{T_hero["hero_subtitle"]}</p>
        <div style="margin-top:1rem;">
            <p style="color:#90EE90; font-size:0.95rem; font-weight:600; margin:0;">{T_hero["hero_analysis"]}</p>
            <p style="color:#C8D8E8; font-size:0.85rem; margin:0.3rem 0;">{T_hero["hero_desc1"]}</p>
            <p style="color:#C8D8E8; font-size:0.85rem; margin:0;">{T_hero["hero_desc2"]}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1rem 0 0.5rem; border-bottom:1px solid rgba(0,212,255,0.15); margin-bottom:1rem; margin-top:0.5rem;">
            <p style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#00D4FF; letter-spacing:0.15em; text-transform:uppercase; margin:0;">{T["filters"]}</p>
        </div>
        """, unsafe_allow_html=True)

        neighbourhoods = ["All"] + sorted(df["neighbourhood_group"].dropna().unique().tolist())
        selected_neighbourhood = st.selectbox(T["neighbourhood"], neighbourhoods)

        # FIX 1: default room type → "Entire home/apt"
        room_types = ["All"] + sorted(df["room_type"].dropna().unique().tolist())
        default_rt_index = room_types.index("Entire home/apt") if "Entire home/apt" in room_types else 0
        selected_room_type = st.selectbox(T["room_type"], room_types, index=default_rt_index)

        top_n = st.slider(T["top_n"], min_value=5, max_value=50, value=10)

        _nb_txt = selected_neighbourhood if selected_neighbourhood != 'All' else 'All Berlin'
        _rt_txt = selected_room_type if selected_room_type != 'All' else 'All types'
        st.markdown(f"""
        <div style="margin-top:2rem; border-top:1px solid rgba(0,212,255,0.15); padding-top:1rem;">
        <p style="font-size:0.7rem; color:#8899AA; font-family:'Space Mono',monospace; margin:0 0 0.3rem;">DATASET</p>
        <p style="font-size:0.9rem; color:#00D4FF; font-family:'Space Mono',monospace; margin:0;">{len(df):,} <span style="color:#8899AA;font-size:0.72rem;">listings</span></p>
        <p style="font-size:0.72rem; color:#8899AA; margin:0.2rem 0 1rem;">Inside Airbnb Berlin</p>
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);border-radius:8px;padding:0.6rem;margin-bottom:0.8rem;">
            <p style="font-size:0.62rem;color:#00D4FF;font-family:'Space Mono',monospace;letter-spacing:0.1em;margin:0 0 0.4rem;">ACTIVE FILTERS</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">📍 {_nb_txt}</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">🏠 {_rt_txt}</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">📊 Top {top_n} listings</p>
        </div>
        <div style="background:rgba(255,215,0,0.05);border:1px solid rgba(255,215,0,0.12);border-radius:8px;padding:0.6rem;">
            <p style="font-size:0.62rem;color:#FFD700;font-family:'Space Mono',monospace;letter-spacing:0.1em;margin:0 0 0.4rem;">RANKING WEIGHTS</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">💰 Price &nbsp;&nbsp;&nbsp; 40%</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">⭐ Reviews &nbsp; 40%</p>
            <p style="font-size:0.78rem;color:#C8D8E8;margin:0.1rem 0;">📈 Demand &nbsp; 20%</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    df_filtered = df.copy()
    if selected_room_type != "All":
        df_filtered = df_filtered[df_filtered["room_type"] == selected_room_type]

    ranked_df = compute_ranking(df_filtered, selected_neighbourhood)
    if ranked_df.empty:
        st.warning("No listings found for the selected filter.")
        st.stop()

    filtered = ranked_df if selected_neighbourhood == "All" else ranked_df[ranked_df["neighbourhood_group"] == selected_neighbourhood]

    _kpi_total = f"{len(filtered):,}"
    _kpi_med = f"€{filtered['price'].median():.0f}"
    _kpi_avg = f"{filtered['number_of_reviews'].mean():.0f}"
    _kpi_top = f"{filtered['ranking_score'].max():.3f}"
    _kpi_night = T['night']
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">{T['total']}</div><div class="kpi-value cyan">{_kpi_total}</div></div>
        <div class="kpi-card"><div class="kpi-label">{T['median']}</div><div class="kpi-value">{_kpi_med}</div><div style="font-size:0.75rem;color:#8899AA;margin-top:0.2rem;">{_kpi_night}</div></div>
        <div class="kpi-card"><div class="kpi-label">{T['avg']}</div><div class="kpi-value">{_kpi_avg}</div></div>
        <div class="kpi-card"><div class="kpi-label">{T['top']}</div><div class="kpi-value gold">{_kpi_top}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">{T["rankings"]}</div>', unsafe_allow_html=True)

    # FIX 2: rename columns and format price/score cleanly
    display_cols = [c for c in ["name", "neighbourhood_group", "room_type", "price", "number_of_reviews", "ranking_score"] if c in ranked_df.columns]
    _tbl = ranked_df[display_cols].head(top_n).reset_index(drop=True)
    _tbl.index = _tbl.index + 1
    _tbl = _tbl.rename(columns={
        "name": "Listing",
        "neighbourhood_group": "Neighbourhood",
        "room_type": "Room Type",
        "price": "Price (€)",
        "number_of_reviews": "Reviews",
        "ranking_score": "Score"
    })
    _tbl["Price (€)"] = _tbl["Price (€)"].apply(lambda x: f"€{int(x)}")

    def _color_score(val):
        try:
            v = float(val)
            if v >= 0.6: return 'color: #00D4FF; font-weight:700'
            elif v >= 0.4: return 'color: #FFD700; font-weight:700'
            return 'color: #FF6B6B; font-weight:700'
        except:
            return ''

    def _hl_top(row):
        style = 'background-color:rgba(0,212,255,0.08);border-left:3px solid #00D4FF' if row.name == 1 else ''
        return [style] * len(row)

    _styled = _tbl.style.applymap(_color_score, subset=['Score']).apply(_hl_top, axis=1)
    st.dataframe(_styled, use_container_width=True, height=400)
    st.caption(T["caption"])

    # Price vs Score scatter
    st.markdown('<div class="section-header">Price vs Score Distribution</div>', unsafe_allow_html=True)
    try:
        import plotly.express as px
        _sc_df = ranked_df[display_cols].head(min(300, len(ranked_df))).copy()
        _name_c = 'name' if 'name' in _sc_df.columns else _sc_df.columns[0]
        _fig_sc = px.scatter(
            _sc_df, x='price', y='ranking_score',
            size='number_of_reviews' if 'number_of_reviews' in _sc_df.columns else None,
            color='ranking_score',
            color_continuous_scale=[[0,'#FF6B6B'],[0.4,'#FFD700'],[0.7,'#00A0CC'],[1,'#00D4FF']],
            hover_name=_name_c,
            hover_data={'price':':.0f','ranking_score':':.3f','number_of_reviews':True,'neighbourhood_group':True},
            labels={'price':'Price (€/night)','ranking_score':'Ranking Score'},
            size_max=28,
        )
        _fig_sc.update_layout(
            paper_bgcolor='#0A1628', plot_bgcolor='#0A1628', font_color='#C8D8E8', height=300,
            margin=dict(l=10,r=10,t=10,b=10), coloraxis_showscale=False,
            xaxis=dict(gridcolor='rgba(0,212,255,0.08)',color='#8899AA',title_font=dict(color='#8899AA')),
            yaxis=dict(gridcolor='rgba(0,212,255,0.08)',color='#8899AA',title_font=dict(color='#8899AA')),
        )
        _fig_sc.update_traces(marker=dict(line=dict(width=0)))
        st.plotly_chart(_fig_sc, use_container_width=True)
    except Exception as _e:
        st.caption(f'Scatter unavailable: {_e}')

    # Review Intelligence
    st.markdown('<div class="section-header">⭐ Review Intelligence</div>', unsafe_allow_html=True)
    try:
        tier_stats, df_review, top_perf, tier_counts = analyze_review_intelligence(filtered)
        if tier_stats is not None:
            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                st.markdown('**Host Tiers**')
                tier_colors = {"Bestseller 🌟": "#FFD700", "Established ⭐": "#00D4FF", "Rising Star 📈": "#90EE90", "New Host 🆕": "#FF6B6B"}
                for tier_name, count in tier_counts.items():
                    color = tier_colors.get(tier_name, "#8899AA")
                    pct = round((count / tier_counts.sum()) * 100, 1)
                    st.markdown(f"""
                    <div style="background:rgba(0,212,255,0.05); border-left:3px solid {color}; padding:0.5rem; margin:0.3rem 0; border-radius:4px;">
                        <p style="color:{color}; font-size:0.9rem; font-weight:600; margin:0;">{tier_name}</p>
                        <p style="color:#8899AA; font-size:0.75rem; margin:0;">{int(count)} hosts ({pct}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            with col_r2:
                st.markdown('**Key Insights**')
                avg_all = filtered['number_of_reviews'].mean()
                bestsellers = df_review[df_review['host_tier'] == "Bestseller 🌟"]
                new_hosts = df_review[df_review['host_tier'] == "New Host 🆕"]
                if not bestsellers.empty:
                    avg_bestseller = bestsellers['number_of_reviews'].mean()
                    median_price_bs = bestsellers['price'].median()
                    st.markdown(f"""
                    <div style="background:rgba(255,215,0,0.1); border:1px solid rgba(255,215,0,0.3); padding:0.7rem; margin:0.3rem 0; border-radius:6px;">
                        <p style="color:#FFD700; font-size:0.85rem; font-weight:600; margin:0;">Bestseller Strategy</p>
                        <p style="color:#C8D8E8; font-size:0.8rem; margin:0.2rem 0;">Avg {int(avg_bestseller)} reviews @ €{int(median_price_bs)}/night</p>
                        <p style="color:#8899AA; font-size:0.75rem; margin:0;">+{int(avg_bestseller - avg_all)} more reviews than average</p>
                    </div>
                    """, unsafe_allow_html=True)
                if not new_hosts.empty:
                    ramp_up = new_hosts['reviews_per_month'].mean()
                    st.markdown(f"""
                    <div style="background:rgba(255,100,100,0.1); border:1px solid rgba(255,100,100,0.3); padding:0.7rem; margin:0.3rem 0; border-radius:6px;">
                        <p style="color:#FF6B6B; font-size:0.85rem; font-weight:600; margin:0;">New Host Ramp-Up</p>
                        <p style="color:#C8D8E8; font-size:0.8rem; margin:0.2rem 0;">Average {ramp_up:.1f} reviews/month to gain traction</p>
                        <p style="color:#8899AA; font-size:0.75rem; margin:0;">~ {int(20/max(ramp_up, 0.1))} months to reach "Rising Star"</p>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.caption(f'Review intelligence unavailable: {e}')

    col_map, col_chart = st.columns([3, 2])
    with col_map:
        st.markdown(f'<div class="section-header">{T["map_title"]}</div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            map_data = (
                df.dropna(subset=["latitude", "longitude", "neighbourhood_group"])
                .groupby("neighbourhood_group")
                .agg(lat=("latitude", "median"), lon=("longitude", "median"),
                     median_price=("price", "median"), listing_count=("price", "count"))
                .reset_index()
            )
            fig_map = px.scatter_mapbox(
                map_data, lat="lat", lon="lon", size="listing_count", color="median_price",
                hover_name="neighbourhood_group",
                hover_data={"median_price": ":€.0f", "listing_count": True, "lat": False, "lon": False},
                color_continuous_scale=[[0, "#0D2137"], [0.4, "#0099CC"], [1, "#00D4FF"]],
                size_max=50, zoom=10, center={"lat": 52.52, "lon": 13.405},
                mapbox_style="carto-darkmatter",
                labels={"median_price": "Median Price (€)", "listing_count": "Listings"},
            )
            fig_map.update_layout(
                paper_bgcolor="#0A1628", margin=dict(l=0, r=0, t=0, b=0), height=430,
                coloraxis_colorbar=dict(title="Price (€)", tickfont=dict(color="#8899AA", size=10),
                                        bgcolor="#0A1628", bordercolor="rgba(0,212,255,0.2)"),
            )
            st.plotly_chart(fig_map, use_container_width=True, key='neighbourhood_map')
        except Exception as e:
            st.caption(f"Map unavailable: {e}")

    with col_chart:
        st.markdown(f'<div class="section-header">{T["chart_title"]}</div>', unsafe_allow_html=True)
        try:
            import plotly.express as px
            price_by_nb = (
                df.groupby("neighbourhood_group")["price"].median().reset_index()
                .sort_values("price", ascending=True)
                .rename(columns={"neighbourhood_group": "Neighbourhood", "price": "Median Price (€)"})
            )
            fig = px.bar(price_by_nb, x="Median Price (€)", y="Neighbourhood", orientation="h",
                         color="Median Price (€)", color_continuous_scale=[[0, "#132140"], [0.5, "#00A0CC"], [1, "#00D4FF"]])
            fig.update_layout(
                paper_bgcolor="#0A1628", plot_bgcolor="#0A1628", font_color="#C8D8E8",
                margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
                xaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#8899AA"),
                yaxis=dict(gridcolor="rgba(0,212,255,0.1)", color="#C8D8E8"), height=430,
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True, key='price_by_neighbourhood')
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")

    st.markdown(f'<div class="section-header">{T["insights_title"]}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button(T["generate"], type="primary")

    if run_button:
        with st.spinner(T["spinner_market"]):
            sample = ranked_df.head(top_n)
            dataset_id = create_langsmith_dataset(sample)
            if dataset_id:
                st.caption(f"✓ Logged to LangSmith · {dataset_id}")
            insights = analyze_listings(ranked_df, selected_neighbourhood)
        display_insights_expander(insights, f"🔍 Market Insights - {selected_neighbourhood}")
        st.toast(T["toast_market"], icon="🤖")

    st.markdown(f'<div class="section-header">{T["analyse_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#132140; border:1px solid rgba(0,212,255,0.15); border-left:3px solid #FFD700; border-radius:0 12px 12px 0; padding:1rem 1.5rem; margin-bottom:1.5rem;">
        <p style="color:#FFD700; font-size:0.8rem; font-family:Space Mono,monospace; letter-spacing:0.1em; margin:0 0 0.3rem;">{T["how_it_works"]}</p>
        <p style="color:#C8D8E8; font-size:0.9rem; margin:0;">{T["how_text"]}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("listing_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            host_price = st.number_input(T["price_label"], min_value=10, max_value=1000, value=85)
        with fc2:
            host_neighbourhood = st.selectbox(T["nb_label"], sorted(df["neighbourhood_group"].dropna().unique().tolist()), key="host_nb")
        with fc3:
            host_reviews = st.number_input(T["reviews_label"], min_value=0, max_value=5000, value=25)
        fc4, fc5 = st.columns(2)
        with fc4:
            host_room_type_form = st.selectbox("Your room type", sorted(df["room_type"].dropna().unique().tolist()), index=sorted(df["room_type"].dropna().unique().tolist()).index("Entire home/apt") if "Entire home/apt" in df["room_type"].dropna().unique().tolist() else 0, key="host_rt")
        with fc5:
            host_amenities = st.text_input(T["amenities_label"], value="balcony, wifi, fully equipped kitchen")
        submitted = st.form_submit_button(T["submit"], type="primary")

    if submitted:
        with st.spinner(T["spinner_listing"]):
            comp_df = compute_ranking(df, host_neighbourhood)

            # Use room type from form for fair comparison
            comp_df_rt = comp_df.copy()
            if "room_type" in comp_df_rt.columns:
                rt_filtered = comp_df_rt[comp_df_rt["room_type"] == host_room_type_form]
                if not rt_filtered.empty:
                    comp_df_rt = rt_filtered

            nb_median = comp_df_rt["price"].median()
            nb_listings = len(comp_df_rt)
            price_score = round(1 - min(host_price / (nb_median * 2), 1), 3)
            max_reviews = comp_df_rt["number_of_reviews"].max()
            review_score = round(host_reviews / max_reviews if max_reviews > 0 else 0, 3)
            demand_score = round(nb_listings / df["neighbourhood_group"].value_counts().max(), 3)
            my_score = round(price_score * 0.4 + review_score * 0.4 + demand_score * 0.2, 3)
            percentile = round((comp_df_rt["ranking_score"] < my_score).mean() * 100, 1)

            selected_listing = pd.Series({
                'price': host_price,
                'number_of_reviews': host_reviews,
                'reviews_per_month': comp_df_rt['reviews_per_month'].mean() * (host_reviews / comp_df_rt['number_of_reviews'].mean()) if comp_df_rt['number_of_reviews'].mean() > 0 else 0.5,
                'availability_365': 200,
                'number_of_reviews_ltm': int(host_reviews * 0.4),
                'last_review': pd.Timestamp.now() - pd.Timedelta(days=15),
                'room_type': host_room_type_form,
                'neighbourhood': host_neighbourhood,
                'name': 'Your Listing'
            })

            recommended_mid, recommended_low, recommended_high = recommend_optimal_price(
                host_price, comp_df_rt, host_room_type=host_room_type_form
            )

            competitors = get_top_competitors(host_price, host_neighbourhood, host_room_type_form, df, comp_df_rt)

            st.session_state.revenue_impact = calculate_revenue_impact(
                host_price,
                (recommended_mid if recommended_mid else host_price),
                selected_listing['reviews_per_month'],
                comp_df_rt['reviews_per_month'].mean(),
                (365 - selected_listing['availability_365']) / 365 * 100
            )
            st.session_state.revenue_impact_ready = True

            personal_prompt = f"""You are a competitive intelligence analyst for the Berlin Airbnb market.
Host listing: price €{host_price}, neighbourhood {host_neighbourhood}, room type {host_room_type_form}, reviews {host_reviews}, amenities: {host_amenities}.
Score: {my_score} (beats {percentile}% of {host_neighbourhood} {host_room_type_form} competitors). Neighbourhood median: €{nb_median:.0f}. Competing listings: {nb_listings}.
Generate exactly 5 specific recommendations: 1) Price positioning vs €{nb_median:.0f} median 2) Review count vs top performers 3) Strongest amenity differentiator 4) One action this week 5) Optimised listing opening sentence. Be direct and use the numbers.
IMPORTANT: Start directly with "*1)" — no preamble, no intro sentence, no meta-commentary."""

            llm = ChatOpenAI(model="gpt-4o-mini")
            response = llm.invoke([HumanMessage(content=personal_prompt)])
            personal_insights = response.content

        if 'revenue_impact_ready' in st.session_state and st.session_state.revenue_impact_ready:
            revenue_impact = st.session_state.revenue_impact
            st.markdown('<div class="section-header">📊 Executive Summary — Your Revenue Opportunity</div>', unsafe_allow_html=True)
            try:
                summ_col1, summ_col2, summ_col3 = st.columns(3)
                with summ_col1:
                    st.markdown(f"""
                    <div style="background:#132140; border:2px solid #8899AA; border-radius:8px; padding:1.2rem; text-align:center;">
                        <p style="color:#8899AA; font-size:0.7rem; text-transform:uppercase; margin:0 0 0.5rem;"><b>Current Annual Revenue</b></p>
                        <p style="color:#00D4FF; font-size:2rem; font-weight:700; margin:0;">€{(host_price * max(selected_listing['reviews_per_month'], 1) * 30 * 12):.0f}</p>
                        <p style="color:#C8D8E8; font-size:0.75rem; margin:0.5rem 0;">At €{host_price}/night</p>
                    </div>
                    """, unsafe_allow_html=True)
                with summ_col2:
                    monthly_impact = revenue_impact['monthly_total_potential']
                    annual_impact = revenue_impact['annual_total_potential']
                    impact_sign = "+" if monthly_impact > 0 else ""
                    impact_color = "#90EE90" if monthly_impact > 0 else "#FF6B6B"
                    st.markdown(f"""
                    <div style="background:#132140; border:2px solid {impact_color}; border-radius:8px; padding:1.2rem; text-align:center;">
                        <p style="color:{impact_color}; font-size:0.7rem; text-transform:uppercase; margin:0 0 0.5rem;"><b>💰 Monthly Opportunity</b></p>
                        <p style="color:{impact_color}; font-size:2rem; font-weight:700; margin:0;">{impact_sign}€{abs(monthly_impact):.0f}</p>
                        <p style="color:#C8D8E8; font-size:0.75rem; margin:0.5rem 0;">= {impact_sign}€{abs(annual_impact):.0f}/year</p>
                    </div>
                    """, unsafe_allow_html=True)
                with summ_col3:
                    potential_annual = (recommended_mid * (selected_listing['reviews_per_month'] * 1.5) * 30 * 12) if recommended_mid else host_price * selected_listing['reviews_per_month'] * 30 * 12
                    st.markdown(f"""
                    <div style="background:#132140; border:2px solid #FFD700; border-radius:8px; padding:1.2rem; text-align:center;">
                        <p style="color:#FFD700; font-size:0.7rem; text-transform:uppercase; margin:0 0 0.5rem;"><b>🎯 Optimized Annual Revenue</b></p>
                        <p style="color:#FFD700; font-size:2rem; font-weight:700; margin:0;">€{potential_annual:.0f}</p>
                        <p style="color:#C8D8E8; font-size:0.75rem; margin:0.5rem 0;">+{((potential_annual / max(host_price * max(selected_listing['reviews_per_month'], 1) * 30 * 12, 1) - 1) * 100):.0f}% upside</p>
                    </div>
                    """, unsafe_allow_html=True)
                if abs(monthly_impact) > 100:
                    insight_type = "⚠️ OPPORTUNITY" if monthly_impact > 0 else "🚨 LOSING MONEY"
                    insight_color = "#90EE90" if monthly_impact > 0 else "#FF6B6B"
                    st.markdown(f"""
                    <div style="background:{insight_color}11; border:1px solid {insight_color}; border-radius:8px; padding:1.5rem; margin-top:1rem;">
                        <p style="color:{insight_color}; font-size:1.1rem; font-weight:700; margin:0 0 0.5rem;">{insight_type}</p>
                        <p style="color:#C8D8E8; font-size:0.95rem; margin:0;">
                            You're currently <b>leaving €{abs(monthly_impact):.0f}/month on the table</b> due to suboptimal pricing
                            and positioning. By implementing the recommendations below, you could earn
                            <span style="color:{insight_color}; font-weight:700;">+€{abs(annual_impact):.0f}/year</span>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Executive summary error: {e}")

        st.markdown('<div class="section-header">📋 Your Action Plan</div>', unsafe_allow_html=True)
        try:
            report_breakdown = create_report_breakdown_saas(
                revenue_impact, recommended_mid, host_price, host_reviews,
                recommended_positioning if 'recommended_positioning' in locals() else "Strategic Positioning",
                positioning_analysis['market_stats'] if 'positioning_analysis' in locals() else {"median_activity": 0.7}
            )
            st.markdown(report_breakdown, unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Actionable report unavailable: {e}")

        if recommended_mid is not None:
            st.markdown('<div class="section-header">💰 Price Recommendation Engine</div>', unsafe_allow_html=True)
            price_col1, price_col2, price_col3, price_col4 = st.columns(4)
            with price_col1:
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(255,100,100,0.3); border-radius:8px; padding:1rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.7rem; margin:0 0 0.4rem;">YOUR PRICE</p>
                    <p style="color:#FF6B6B; font-size:2rem; font-weight:700; margin:0;">€{int(host_price)}</p></div>""", unsafe_allow_html=True)
            with price_col2:
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(0,212,255,0.3); border-radius:8px; padding:1rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.7rem; margin:0 0 0.4rem;">RECOMMENDED</p>
                    <p style="color:#00D4FF; font-size:2rem; font-weight:700; margin:0;">€{int(recommended_mid)}</p></div>""", unsafe_allow_html=True)
            with price_col3:
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(100,200,100,0.3); border-radius:8px; padding:1rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.7rem; margin:0 0 0.4rem;">SWEET SPOT RANGE</p>
                    <p style="color:#90EE90; font-size:1.3rem; font-weight:700; margin:0;">€{int(recommended_low)}-€{int(recommended_high)}</p></div>""", unsafe_allow_html=True)
            with price_col4:
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(255,215,0,0.3); border-radius:8px; padding:1rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.7rem; margin:0 0 0.4rem;">MARKET MEDIAN</p>
                    <p style="color:#FFD700; font-size:2rem; font-weight:700; margin:0;">€{int(nb_median)}</p></div>""", unsafe_allow_html=True)
            st.write("")
            if host_price < recommended_low:
                st.info(f"💡 **Opportunity**: Your price (€{int(host_price)}) is below the optimal range (€{int(recommended_low)}-€{int(recommended_high)}). Consider increasing to match demand.")
            elif host_price > recommended_high:
                st.warning(f"⚠️ **Risk**: Your price (€{int(host_price)}) exceeds the sweet spot (€{int(recommended_low)}-€{int(recommended_high)}). You may be pricing yourself out of competition.")
            else:
                st.success(f"✅ **Strong**: Your price (€{int(host_price)}) is well-positioned in the optimal range (€{int(recommended_low)}-€{int(recommended_high)}).")

        # FIX 5: Smart Alerts — removed "Weekly Alerts coming soon" block
        st.markdown('<div class="section-header">🔔 Smart Alerts</div>', unsafe_allow_html=True)
        try:
            alerts = []
            price_diff_pct = abs(host_price - recommended_mid) / recommended_mid * 100 if recommended_mid else 0
            if price_diff_pct > 15:
                if host_price < recommended_mid:
                    alerts.append({"type": "opportunity", "title": "💰 Pricing Opportunity",
                        "message": f"You're underpricing by {price_diff_pct:.0f}%. Comparable listings at €{int(recommended_mid)} average {comp_df_rt['reviews_per_month'].mean():.2f} reviews/month."})
                else:
                    alerts.append({"type": "warning", "title": "⚠️ Price Risk",
                        "message": f"You're overpricing by {price_diff_pct:.0f}%. Consider €{int(recommended_low)}-€{int(recommended_high)} to stay competitive."})
            if selected_listing['reviews_per_month'] < comp_df_rt['reviews_per_month'].median() * 0.5:
                alerts.append({"type": "info", "title": "📊 Low Booking Activity",
                    "message": f"Your booking rate ({selected_listing['reviews_per_month']:.2f}/month) is below market median ({comp_df_rt['reviews_per_month'].median():.2f}/month). Try updating photos or opening more availability."})
            if host_reviews < comp_df_rt['number_of_reviews'].median() * 0.3:
                alerts.append({"type": "info", "title": "🔄 Build Credibility",
                    "message": f"Consider promotional pricing to build reviews. Current: {host_reviews} vs median: {int(comp_df_rt['number_of_reviews'].median())}"})
            if alerts:
                for alert in alerts:
                    if alert["type"] == "opportunity": st.success(f"**{alert['title']}**: {alert['message']}")
                    elif alert["type"] == "warning": st.warning(f"**{alert['title']}**: {alert['message']}")
                    else: st.info(f"**{alert['title']}**: {alert['message']}")
            else:
                st.success("✅ **Perfect!** Your listing is well-positioned. Monitor weekly for market changes.")
        except Exception as e:
            st.caption(f"Alerts unavailable: {e}")

        # Positioning Strategy
        st.markdown('<div class="section-header">🎯 Positioning Strategy</div>', unsafe_allow_html=True)
        try:
            positioning_analysis = analyze_positioning_opportunity(df, host_neighbourhood, selected_room_type)
            if positioning_analysis:
                recommended_positioning, recommendation_reason = get_positioning_recommendation(
                    host_price, selected_listing['reviews_per_month'], positioning_analysis['market_stats']
                )
                st.markdown(f"""
                <div style="background:rgba(0,212,255,0.1); border:2px solid #00D4FF; border-radius:8px; padding:1.2rem;">
                    <p style="color:#8899AA; font-size:0.75rem; margin:0 0 0.5rem; text-transform:uppercase;"><b>Market Recommendation</b></p>
                    <p style="color:#00D4FF; font-size:1.4rem; font-weight:700; margin:0 0 0.5rem;">{recommended_positioning}</p>
                    <p style="color:#C8D8E8; font-size:0.9rem; margin:0;">{recommendation_reason}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Positioning analysis unavailable: {e}")

        # Top 3 Competitors
        if not competitors.empty:
            st.markdown('<div class="section-header">🏆 Top 3 Direct Competitors</div>', unsafe_allow_html=True)
            for idx, comp_row in competitors.iterrows():
                comp_name = comp_row.get("name", "N/A")
                comp_price = int(comp_row.get("price", 0))
                comp_reviews = int(comp_row.get("number_of_reviews", 0))
                comp_score = comp_row.get("ranking_score", 0)
                comp_neighbourhood = comp_row.get("neighbourhood_group", "N/A")
                # FIX 3: clean price diff display
                price_diff = comp_price - host_price
                if price_diff > 0:
                    price_indicator = f"<span style='color:#90EE90;'>(+€{price_diff} vs yours)</span>"
                elif price_diff < 0:
                    price_indicator = f"<span style='color:#FF6B6B;'>(-€{abs(price_diff)} vs yours)</span>"
                else:
                    price_indicator = f"<span style='color:#FFD700;'>(same price)</span>"
                score_color = "#00D4FF" if comp_score >= 0.6 else "#FFD700" if comp_score >= 0.4 else "#FF6B6B"
                st.markdown(f"""
                <div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:1rem; margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <p style="color:#C8D8E8; font-size:0.95rem; font-weight:600; margin:0;">#{idx+1} {comp_name[:40]}</p>
                        <p style="color:{score_color}; font-size:0.9rem; font-weight:700;">Score: {comp_score:.3f}</p>
                    </div>
                    <div style="display:flex; gap:2rem; font-size:0.85rem;">
                        <div><span style="color:#8899AA;">Price:</span> <span style="color:#C8D8E8; font-weight:600;">€{comp_price}</span> {price_indicator}</div>
                        <div><span style="color:#8899AA;">Reviews:</span> <span style="color:#C8D8E8; font-weight:600;">{comp_reviews}</span></div>
                        <div><span style="color:#8899AA;">Zone:</span> <span style="color:#C8D8E8; font-weight:600;">{comp_neighbourhood}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No similar competitors found in this price range and room type.")

        # Revenue Simulator — FIX 4: hide_index=True
        st.markdown('<div class="section-header">💵 Revenue Optimization Simulator</div>', unsafe_allow_html=True)
        try:
            scenarios = calculate_revenue_scenarios(host_price, comp_df_rt, nb_median)
            current = next((s for s in scenarios if s['is_current']), scenarios[2])
            col_sim1, col_sim2 = st.columns([2, 1])
            with col_sim1:
                display_scenarios = [{k: v for k, v in s.items() if k != 'is_current'} for s in scenarios]
                st.dataframe(pd.DataFrame(display_scenarios), use_container_width=True, height=200, hide_index=True)
            with col_sim2:
                st.markdown(f"""
                <div style="background:#132140; border:1px solid rgba(0,212,255,0.3); border-radius:8px; padding:1rem;">
                    <p style="color:#8899AA; font-size:0.7rem; margin:0 0 0.3rem;">YOUR CURRENT SCENARIO</p>
                    <p style="color:#00D4FF; font-size:1.8rem; font-weight:700; margin:0;">{current['Monthly']}</p>
                    <p style="color:#8899AA; font-size:0.75rem; margin:0.3rem 0;">monthly revenue</p>
                    <p style="color:#C8D8E8; font-size:0.8rem; margin:0.3rem 0;">📊 {current['Occupancy']} occupancy</p>
                    <p style="color:#C8D8E8; font-size:0.8rem; margin:0;">🏆 {current['Position']}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Simulator unavailable: {e}")

        # KPI
        st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
        try:
            kpi_data = calculate_kpi_metrics(selected_listing)
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                occ_pct = kpi_data["occupancy_rate"]
                occ_color = "#90EE90" if occ_pct >= 60 else "#FFD700" if occ_pct >= 40 else "#FF6B6B"
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:0.8rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.65rem; margin:0 0 0.3rem;">OCCUPANCY RATE</p>
                    <p style="color:{occ_color}; font-size:1.8rem; font-weight:700; margin:0;">{occ_pct:.1f}%</p></div>""", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:0.8rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.65rem; margin:0 0 0.3rem;">ADR (Avg Daily Rate)</p>
                    <p style="color:#00D4FF; font-size:1.8rem; font-weight:700; margin:0;">€{kpi_data['adr']:.0f}</p></div>""", unsafe_allow_html=True)
            with kpi3:
                revpar_val = kpi_data["revpar"]
                revpar_color = "#90EE90" if revpar_val >= 60 else "#FFD700" if revpar_val >= 30 else "#FF6B6B"
                st.markdown(f"""<div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:0.8rem; text-align:center;">
                    <p style="color:#8899AA; font-size:0.65rem; margin:0 0 0.3rem;">RevPAR</p>
                    <p style="color:{revpar_color}; font-size:1.8rem; font-weight:700; margin:0;">€{revpar_val:.0f}</p></div>""", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"KPI calculation unavailable: {e}")

        # Review Quality
        st.markdown('<div class="section-header">⭐ Review Quality Analytics</div>', unsafe_allow_html=True)
        try:
            # Estimate host reviews_per_month from actual reviews (assume ~24 months active)
            host_rpm_estimate = host_reviews / 24.0
            quality_metrics = calculate_review_quality(host_reviews, host_rpm_estimate, comp_df_rt)
            col_qual1, col_qual2, col_qual3, col_qual4 = st.columns(4)
            metrics = [
                ("VELOCITY", f"{quality_metrics['velocity_score']*100:.0f}%", "#90EE90" if quality_metrics['velocity_score'] >= 0.7 else "#FFD700" if quality_metrics['velocity_score'] >= 0.4 else "#FF6B6B"),
                ("CONSISTENCY", f"{quality_metrics['consistency_score']*100:.0f}%", "#90EE90" if quality_metrics['consistency_score'] >= 0.85 else "#FFD700" if quality_metrics['consistency_score'] >= 0.6 else "#FF6B6B"),
                ("RECENCY", f"{quality_metrics['recency_score']*100:.0f}%", "#90EE90" if quality_metrics['recency_score'] >= 0.7 else "#FFD700" if quality_metrics['recency_score'] >= 0.4 else "#FF6B6B"),
                ("OVERALL", f"{quality_metrics['quality_score']*100:.0f}%", "#00D4FF" if quality_metrics['quality_score'] >= 0.8 else "#FFD700" if quality_metrics['quality_score'] >= 0.6 else "#FF6B6B"),
            ]
            for col, (label, val, color) in zip([col_qual1, col_qual2, col_qual3, col_qual4], metrics):
                with col:
                    st.markdown(f"""<div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:8px; padding:0.8rem; text-align:center;">
                        <p style="color:#8899AA; font-size:0.65rem; margin:0 0 0.3rem;">{label}</p>
                        <p style="color:{color}; font-size:1.6rem; font-weight:700; margin:0;">{val}</p></div>""", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Quality analytics unavailable: {e}")

        col_score, col_insights = st.columns([1, 2])
        with col_score:
            score_color = "#00D4FF" if percentile >= 50 else "#FFD700" if percentile >= 25 else "#FF6B6B"
            st.markdown(f"""
            <div style="background:#132140; border:1px solid rgba(0,212,255,0.2); border-radius:12px; padding:1.5rem; text-align:center;">
                <p style="color:#8899AA; font-size:0.7rem; font-family:Space Mono,monospace; letter-spacing:0.1em; margin:0 0 0.5rem;">{T["your_score"]}</p>
                <p style="color:{score_color}; font-size:3rem; font-weight:700; margin:0; line-height:1;">{my_score}</p>
                <p style="color:#8899AA; font-size:0.85rem; margin:0.5rem 0 1.5rem;">{T["beats"]} {percentile}% of {host_neighbourhood}</p>
                <div style="border-top:1px solid rgba(0,212,255,0.1); padding-top:1rem;">
                    <p style="color:#8899AA; font-size:0.7rem; font-family:Space Mono,monospace; margin:0 0 0.3rem;">{T["breakdown"]}</p>
                    <p style="color:#C8D8E8; font-size:0.8rem; margin:0.2rem 0;">Price &nbsp; {price_score:.3f} x 40%</p>
                    <p style="color:#C8D8E8; font-size:0.8rem; margin:0.2rem 0;">Reviews &nbsp; {review_score:.3f} x 40%</p>
                    <p style="color:#C8D8E8; font-size:0.8rem; margin:0.2rem 0;">Demand &nbsp; {demand_score:.3f} x 20%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_insights:
            display_insights_expander(personal_insights, "🎯 Your Personalized Insights")
        st.toast(T["toast_listing"], icon="🏠")

    st.markdown("""
    <div class="footer">
        BerlinHostAIQ &nbsp;·&nbsp; Antonio Ciraci &nbsp;·&nbsp; AI Bootcamp Final Project &nbsp;·&nbsp; March 2026
        &nbsp;·&nbsp; Data: Inside Airbnb &nbsp;·&nbsp; Model: GPT-4o-mini &nbsp;·&nbsp; Tracing: LangSmith
        &nbsp;·&nbsp; <a href="https://github.com/ciraci2-netizen/final-project-antonio-ciraci" style="color:#00D4FF; text-decoration:none;">GitHub &#8599;</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        print("Loading dataset...")
        df = load_data()
        print(f"✓ {len(df):,} listings loaded")
        ranked = compute_ranking(df)
        print(f"✓ Top: {ranked.iloc[0].get('name', 'N/A')} — {ranked.iloc[0]['ranking_score']}")
        create_langsmith_dataset(ranked.head(20))
        insights = analyze_listings(ranked)
        print(insights)

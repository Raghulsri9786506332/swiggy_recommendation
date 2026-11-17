import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from recommend import get_recommendations
from utils import get_unique_cities, get_unique_cuisines

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------

st.markdown("""
<h1 style='text-align:center;color:#FC8019;'>
🍽️ Swiggy Restaurant Recommender
</h1>

<p style='text-align:center;color:gray;'>
Discover restaurants with K-Means clustering and your preferences
</p>

<hr>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

cities = []
cuisines = []

try:
    cities = get_unique_cities()
    cuisines = get_unique_cuisines()

except FileNotFoundError:
    st.error(
        "❌ Data files not found. Run:\n\npython src/preprocess.py"
    )
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

st.sidebar.header("🔍 Your Preferences")

selected_city = st.sidebar.selectbox(
    "📍 Select City",
    cities
)

selected_cuisine = st.sidebar.selectbox(
    "🍜 Select Cuisine",
    cuisines
)

min_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=1.0,
    max_value=5.0,
    value=3.5,
    step=0.1
)

max_cost = st.sidebar.slider(
    "💰 Maximum Cost (₹)",
    min_value=50,
    max_value=2000,
    value=500,
    step=50
)

rating_count = st.sidebar.number_input(
    "👥 Minimum Ratings Count",
    min_value=0,
    value=100
)

top_n = st.sidebar.slider(
    "📋 Number of Recommendations",
    min_value=3,
    max_value=100,
    value=10
)

search_button = st.sidebar.button(
    "🔎 Find Restaurants",
    use_container_width=True
)

# -----------------------------------------------------------------------------
# RECOMMENDATION SECTION
# -----------------------------------------------------------------------------

if search_button:

    user_input = {
        "city": selected_city,
        "cuisine": selected_cuisine,
        "rating": min_rating,
        "rating_count": rating_count,
        "cost": max_cost
    }

    with st.spinner("Finding restaurants for you..."):
        results = get_recommendations(
            user_input=user_input,
            top_n=top_n
        )

    if results.empty:
        st.warning(
            "⚠️ No restaurants found. Try changing your filters."
        )

    else:

        st.success(
            f"✅ Found {len(results)} matching restaurants!"
        )

        st.info(
            f"📍 {selected_city} | 🍽️ {selected_cuisine} | ⭐ {min_rating}+ Rating"
        )

        st.markdown("---")

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1
        ):

            if rank == 1:
                badge = "🥇"
            elif rank == 2:
                badge = "🥈"
            elif rank == 3:
                badge = "🥉"
            else:
                badge = "🍴"

            cost = float(row["cost"])

            if cost < 300:
                cost_category = "Budget"
            elif cost < 700:
                cost_category = "Moderate"
            else:
                cost_category = "Premium"

            with st.container():

                col1, col2, col3 = st.columns([4, 1, 1])

                with col1:

                    st.markdown(
                        f"### {badge} {rank}. {row['name']}"
                    )

                    if "address" in row:
                        st.write(f"📍 {row['address']}")

                    if "cuisine" in row:
                        st.write(f"🍽️ {row['cuisine']}")

                    st.write(
                        f"💵 Category: {cost_category}"
                    )

                with col2:

                    st.metric(
                        "⭐ Rating",
                        row["rating"]
                    )

                    st.metric(
                        "💰 Cost",
                        f"₹{int(cost)}"
                    )

                with col3:

                    if "similarity_score" in row:
                        st.metric(
                            "Match Score",
                            f"{row['similarity_score']:.2%}"
                        )

                    if "link" in row and pd.notna(row["link"]):
                        st.markdown(
                            f"[🔗 View on Swiggy]({row['link']})"
                        )

                st.markdown("---")

        # ---------------------------------------------------------------------
        # DOWNLOAD CSV
        # ---------------------------------------------------------------------

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Recommendations",
            data=csv,
            file_name="restaurant_recommendations.csv",
            mime="text/csv"
        )

        # ---------------------------------------------------------------------
        # TABLE VIEW
        # ---------------------------------------------------------------------

        with st.expander("📊 View as Table"):

            display_columns = [
                col for col in [
                    "name",
                    "city",
                    "cuisine",
                    "rating",
                    "cost",
                    "address",
                    "similarity_score"
                ]
                if col in results.columns
            ]

            st.dataframe(
                results[display_columns],
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# LANDING PAGE
# -----------------------------------------------------------------------------

else:

    st.markdown("""
## 👋 Welcome!

This recommendation system helps you discover restaurants using:

- 🍽️ Cuisine Preferences
- ⭐ Restaurant Ratings
- 💰 Budget Constraints
- 📍 City-Based Filtering
- 🤖 Similarity-Based Recommendation Engine

### How to Use

1. Select a city
2. Choose a cuisine
3. Set rating and budget filters
4. Click **Find Restaurants**
5. Get personalized recommendations instantly
""")
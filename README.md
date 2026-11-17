# 🍽️ Swiggy Restaurant Recommendation System

A machine learning-powered recommendation system that clusters restaurants with K-means and recommends options that match user preferences.

## 📋 Project Overview

This system analyzes Swiggy restaurant data and provides personalized recommendations based on:
- **Location** (City)
- **Cuisine Type** (Biryani, North Indian, South Indian, etc.)
- **Rating Threshold** (1.0 - 5.0 stars)
- **Budget** (Cost per person in ₹)
- **Popularity** (Minimum number of ratings)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

1. **Open a terminal in this project folder** (the folder containing `app.py`).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data preprocessing:**
   ```bash
   python src/preprocess.py
   ```
   This generates:
   - `data/cleaned_data.csv` — Cleaned dataset without duplicates/missing values
   - `data/encoded_data.csv` — One-Hot encoded features for clustering
   - `models/encoder.pkl` — Serialized encoder for Streamlit

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
swiggy_recommendation/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── src/                  # Core modules
│   ├── preprocess.py     # Data cleaning & One-Hot encoding
│   ├── recommend.py      # K-means clustering recommendation engine
│   └── utils.py          # Utility functions for UI
│
├── data/                 # Data files
│   ├── swiggy.csv           # Raw restaurant data (user-provided)
│   ├── cleaned_data.csv     # Generated: cleaned dataset
│   └── encoded_data.csv     # Generated: encoded features
│
└── models/               # Trained artifacts
    └── encoder.pkl       # Generated: OneHotEncoder
```

## 🔄 System Workflow

### Step 1: Data Preprocessing (`src/preprocess.py`)

**Purpose:** Clean raw data and prepare numerical features

**Operations:**
- ✅ Remove duplicate restaurant entries
- ✅ Clean data quality issues:
  - Replace '--' ratings with NaN
  - Parse rating counts (e.g., "500+ ratings" → 500)
  - Remove ₹ symbol and commas from cost
- ✅ Handle missing values:
  - Drop rows with missing critical fields
  - Fill rating_count with median
- ✅ One-Hot encode categorical features (city, cuisine)

**Input:** `data/swiggy.csv`

**Outputs:**
- `data/cleaned_data.csv` — Original data with cleaning applied, no encoding
- `data/encoded_data.csv` — Numerical dataset (41 columns for the included data)
- `models/encoder.pkl` — Pickle file to transform user input

**Data Schema:**
| Column | Type | Example |
|--------|------|---------|
| id | int | 1 |
| name | str | "Biryani House" |
| city | str | "Bangalore" |
| rating | float | 4.5 |
| rating_count | float | 500 |
| cost | float | 250 |
| cuisine | str | "Biryani" |
| lic_no | str | "LIC001" |
| link | str | "https://swiggy.com/1" |
| address | str | "HSR Layout" |
| menu | str | "Biryani Menu" |

### Step 2: Recommendation Engine (`src/recommend.py`)

**Algorithm:** K-means clustering with distance-based ranking

**How it works:**
1. Loads the cleaned data, aligned encoded data, and saved OneHotEncoder.
2. Standardizes the three numeric features so cost and rating counts do not dominate clustering.
3. Fits a reproducible K-means model (`random_state=42`), using up to 8 clusters (fewer when the dataset has fewer distinct rows).
4. Encodes and scales the user's preferences, then assigns them to the nearest cluster.
5. Applies the selected city, cuisine, rating, cost, and popularity filters to the original rows.
6. Ranks eligible restaurants in the assigned cluster by distance from the user preferences. If that cluster has no eligible restaurants, it ranks all eligible rows instead.
7. Maps the selected row indices back to `cleaned_data.csv` for display.

The displayed match score is an inverse-distance ranking score (`1 / (1 + distance)`), not a probability.

**Key Functions:**
```python
load_data()              # Load aligned data and fit/cache the K-means model
build_user_vector()      # Encode and scale user preferences
get_recommendations()    # Return top-N eligible restaurants from the assigned cluster
```

**Example Calculation:**
```
User Input: {city: "Bangalore", cuisine: "Biryani", rating: 4.5, cost: 250, rating_count: 300}
                              ↓
            One-Hot Encode + Scale numeric features
                              ↓
            [numerical features] + [city encoding] + [cuisine encoding]
                              ↓
            Assign to the nearest K-means cluster
                              ↓
            Rank eligible restaurants in that cluster by distance
```

### Step 3: Streamlit Application (`app.py`)

**Features:**
- 🎨 Beautiful Swiggy-themed UI
- 🔍 Interactive sidebar filters
- 📊 Real-time recommendation display
- ⚡ Card-based and table views
- 🔗 Direct Swiggy links

**Components:**

**Sidebar Input Panel:**
- City selector (dropdown with unique cities)
- Cuisine selector (dropdown with unique cuisines)
- Rating slider (1.0 - 5.0 stars, default 3.5)
- Cost slider (₹50 - ₹2000, default ₹500)
- Min ratings count (number input, default 100)
- Number of recommendations slider (3-100, default 10)
- "Find Restaurants" button

**Main Display Area:**
- Landing page with instructions (no search yet)
- Recommendation cards (after search):
  - Restaurant name & ranking
  - Address and cuisine
  - Rating and cost metrics
  - Match score (%)
  - Swiggy link
- Expandable detailed table

## 💡 How K-means Recommendations Work

The encoded restaurant features are clustered with K-means. The numeric columns
(`rating`, `rating_count`, and `cost`) are standardized before clustering, while
one-hot encoded city and cuisine values remain categorical indicators.

### Feature Encoding Example

```
Original Data:
┌─────────┬────────┬──────────┐
│ rating  │ city   │ cuisine  │
├─────────┼────────┼──────────┤
│ 4.5     │ Mumbai │ Biryani  │
│ 4.2     │ Delhi  │ Chinese  │
└─────────┴────────┴──────────┘

After One-Hot Encoding:
┌─────────┬────────────┬────────────┬─────────────┬─────────────┬────────────┐
│ rating  │ city_Delhi │ city_Mumbai │ cuisine_Biryan │ cuisine_Chinese │
├─────────┼────────────┼────────────┼─────────────┼─────────────┼────────────┤
│ 4.5     │ 0          │ 1          │ 1           │ 0           │
│ 4.2     │ 1          │ 0          │ 0           │ 1           │
└─────────┴────────────┴────────────┴─────────────┴─────────────┴────────────┘
```

The user's preferences are encoded and scaled in the same way, then assigned to
the nearest cluster center. Eligible restaurants from that cluster are ranked
by Euclidean distance to the preference vector. A smaller distance means a
closer match; the displayed match score is `1 / (1 + distance)`.

## 📊 Dataset and Clustering Check

The included dataset (`data/swiggy.csv`) currently contains:
- **3,600 restaurants** across **20 cities**
- **18 cuisine types**
- **Rating range:** 3.8 - 4.9 stars
- **Cost range:** ₹60 - ₹750 per person
- **Popularity range:** 50 - 2,490 ratings

After preprocessing:
- **0 duplicates** removed
- **0 rows dropped** (all data valid)
- **41 encoded columns** (3 numerical + 38 one-hot encoded)
- **3,600 rows** in both cleaned and encoded datasets

With the current features and 8 clusters, the sampled silhouette score
(500 rows, random seed 42) is approximately **0.10**. This indicates overlapping
groups, so interpret K-means clusters as broad recommendation neighborhoods,
not sharply separated restaurant segments.




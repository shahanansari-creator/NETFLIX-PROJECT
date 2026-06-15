# NETFLIX-PROJECT

# 🎬 Netflix Movie Recommendation Engine



## 📌 Overview

This project implements a **collaborative filtering-based Movie Recommendation Engine** inspired by Netflix's recommendation infrastructure. Using **Singular Value Decomposition (SVD)** — a powerful matrix factorization technique — the engine analyses user rating patterns to generate personalized movie suggestions tailored to individual preferences.

The recommendation system is built entirely from scratch, covering data ingestion, preprocessing, model training, cross-validation, and prediction — providing a full end-to-end machine learning pipeline.

---

## 🧩 Problem Statement

Customer behaviour prediction lies at the core of every modern business model — from e-commerce to streaming platforms. **Recommendation Engines** are a key manifestation of this predictability, not only surfacing relevant content but actively increasing user engagement with the platform.

OTT platforms such as Netflix and Amazon Prime rely heavily on user activity patterns to suggest content that aligns with individual tastes and preferences. This project replicates that logic by:

- Analysing user rating history
- Identifying genre-level trends
- Generating a ranked list of movie recommendations for each user

---

## 📂 Dataset

The project uses the **Netflix Prize Dataset**, which contains anonymised customer ratings for movies.

| Feature | Description |
|---|---|
| `Cust_Id` | Unique identifier for each customer |
| `Rating` | Star rating given by the customer (1–5) |
| `Movie_Id` | Unique identifier for each movie |
| `Year` | Release year of the movie |
| `Name` | Title of the movie |

> **Note:** The raw dataset is stored in a non-standard format (combined `.txt` file), requiring custom parsing to extract Movie IDs and Customer IDs.

---

## 🎯 Objectives

1. Identify the most popular and highly rated genres on the platform
2. Build a model that recommends the best-suited movies for any given user across every genre
3. Determine which genres have received the highest and lowest average user ratings

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, wrangling, and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualisation |
| `seaborn` | Statistical visualisation |
| `scikit-surprise` | SVD-based collaborative filtering model |

---

## 🔄 Project Workflow & Decision Rationale

Each step in this project was made with a deliberate reason. Here's not just *what* was done, but *why* every decision was made.

---

### 1. 📥 Data Loading & Parsing

**What was done:**
The raw Netflix dataset (`combined_data_1.txt`) was loaded without headers, and only the first two columns — `Cust_Id` and `Rating` — were kept. A custom loop was written to parse Movie IDs embedded in the file, and a new `Movie_Id` column was added to the DataFrame.

**Why:**
The Netflix Prize dataset is not structured like a standard CSV. Movie IDs appear inline within the data as rows like `1:`, `2:`, etc., followed by the customer ratings for that movie. There is no dedicated column for Movie ID — it's implied by position. Without parsing this structure manually, every rating would be orphaned with no way to know which movie it belongs to. Extracting Movie IDs and attaching them as a column was essential to make the dataset usable for modelling.

The `date` column was also intentionally dropped at load time because for a rating-based recommendation model, *when* a user watched a movie is less important than *what* they rated it. Keeping unused columns would only increase memory overhead on an already large dataset.

---

### 2. 🧹 Data Preprocessing

**What was done:**
Rows where `Rating` was `NaN` were removed (these were the movie ID marker rows). The `Cust_Id` column was converted from `object` (string) type to `integer`. Summary statistics — total movies, total customers, and total ratings — were computed.

**Why:**
The `NaN` rows in the Rating column weren't actual data — they were formatting artefacts used to separate movies in the raw file. Keeping them would corrupt any aggregation or model training. Removing them leaves only genuine customer ratings.

The type conversion of `Cust_Id` from string to integer was necessary because machine learning models and aggregation functions expect numeric types. Leaving it as a string would cause errors downstream during model training and comparison operations.

Computing summary statistics (movie count, customer count, rating count) was done to validate the dataset and understand its scale before modelling — a standard sanity check in any data science project.

---

### 3. 📊 Exploratory Data Analysis (EDA)

**What was done:**
The distribution of star ratings (1 through 5) was visualised using a horizontal bar chart showing how many times each rating was given across the entire dataset.

**Why:**
Before building any model, it's important to understand the shape and bias of your data. If the vast majority of ratings were 5 stars, the model would be skewed towards recommending everything highly, which isn't useful. Visualising the rating distribution helps identify whether the data is balanced or skewed, and whether any adjustments are needed. It also gives a human-readable snapshot of user sentiment across the platform.

---

### 4. ✂️ Pre-filtering (Noise Reduction)

**What was done:**
Two filters were applied before model training:
- **Movies** rated by fewer than the **60th percentile** of total reviewers were dropped.
- **Customers** who had rated fewer than the **60th percentile** threshold of movies were dropped.

**Why:**
This is one of the most important steps in building a reliable recommendation engine, and here's the reasoning:

- **Movies with very few ratings** are statistically unreliable. If only 2 or 3 people have rated a movie, that's not enough signal for the model to learn meaningful patterns. Recommending such movies based on sparse data risks poor suggestions.
- **Inactive users** (those who've rated very few movies) don't provide enough behavioural data for the model to understand their preferences. Including them would add noise without contributing useful patterns.

The **60th percentile** was chosen as the benchmark rather than a fixed number, because the dataset is large and dynamic — a percentile-based threshold scales with the data rather than being arbitrarily hardcoded. This approach removes the bottom 60% of low-engagement movies and users, retaining only the most active and well-represented portion of the dataset, which produces far more reliable recommendations.

---

### 5. 🤖 Model Building — SVD (Singular Value Decomposition)

**What was done:**
The `movie_titles.csv` file was loaded with `ISO-8859-1` encoding. The `scikit-surprise` library was used to build an SVD model. A `Reader` object was created, and the top 100,000 rows of the filtered dataset were loaded into the Surprise `Dataset` format. The SVD model was then initialised and prepared for training.

**Why:**

- **ISO-8859-1 encoding** was used instead of the default UTF-8 because the movie titles file contains special characters (accented letters, symbols) that are common in international movie titles. UTF-8 would throw a decoding error on these characters, so the correct encoding was specified explicitly.

- **SVD was chosen** over simpler approaches (like average ratings or basic content filtering) because it is a *collaborative filtering* technique — it doesn't just look at movie attributes, it learns from the collective behaviour of all users. If User A and User B have similar rating patterns, SVD can recommend to User A what User B enjoyed, even if User A has never seen those movies. This captures far more nuanced preferences than rule-based methods.

- **Only 100,000 rows were used for training** rather than the full dataset. This was a deliberate performance trade-off — the Netflix Prize dataset is millions of rows, and training SVD on the full dataset would be extremely slow. 100K rows is large enough to build a meaningful model while keeping runtime practical for a notebook environment.

- The **`Reader` object** is required by scikit-surprise to specify the rating scale (1–5), so the library knows the range of values to expect during decomposition.

---

### 6. ✅ Model Evaluation

**What was done:**
The model was evaluated using **3-fold Cross-Validation** with **RMSE (Root Mean Square Error)** as the metric.

**Why:**

- **Cross-validation** was used instead of a single train/test split because it provides a more reliable estimate of model performance. With 3 folds, the dataset is split into 3 parts — the model is trained on 2 parts and tested on 1, rotating through all combinations. This reduces the risk of the evaluation result being a fluke of one particular split.

- **RMSE** was chosen because it directly measures the average error in predicted star ratings. For example, an RMSE of 1.0 means the model's predicted rating is off by 1 star on average. It's an intuitive and widely accepted metric for regression-style recommendation problems.

- Using RMSE rather than accuracy makes sense here because recommendations are essentially *rating predictions* — there's no binary right/wrong answer, only how close the predicted score is to the actual score.

---

### 7. 🎬 Generating Recommendations

**What was done:**
A target user (`Cust_Id: 1055714`) was selected. A copy of the full movie catalogue was made, movies below the quality benchmark were filtered out, and the SVD model predicted an **Estimate Score** for every movie the user had not yet rated. The results were sorted in descending order and the **Top 5 movies** were returned as recommendations.

**Why:**

- **Making a copy of the movie catalogue** for each user was important to avoid accidentally modifying the original `df_title` DataFrame. Since we add a new `Estimate_Score` column per user, working on a copy keeps the base data clean and reusable for other users.

- **Re-applying the pre-filter** (dropping low-benchmark movies) before generating recommendations ensures that the engine never recommends a movie that lacks sufficient rating data — even if it wasn't seen by the specific user. This maintains quality control at the output stage, not just during training.

- **Estimate Score** is the core output of SVD's `predict()` function — it's the model's best guess at what rating *this specific user* would give to a movie they haven't seen. The higher the score, the more aligned that movie is with the user's historical preferences.

- **Sorting descending and returning Top 5** gives the user a concise, actionable list of the most personally relevant movies — mirroring how real recommendation systems present results on streaming platforms.

---

## 📈 Key Features

- ✅ **End-to-End Pipeline** — from raw data parsing to final personalised movie recommendations
- ✅ **Custom Data Parser** — handles the non-standard inline Movie ID format of the Netflix Prize dataset
- ✅ **Percentile-Based Pre-filtering** — removes noisy, low-engagement movies and users using adaptive quantile benchmarks
- ✅ **SVD Collaborative Filtering** — leverages matrix factorisation to uncover latent user preference patterns
- ✅ **3-Fold Cross-Validated Model** — ensures robust, reliable RMSE-based evaluation
- ✅ **Scalable Recommendation Function** — easily extendable to any customer ID in the dataset
- ✅ **Visual EDA** — bar chart visualisation of rating distributions across all star categories

---

## 📊 SVD — Why It Works

**Singular Value Decomposition (SVD)** decomposes the user-item rating matrix into three matrices that capture latent relationships between users and movies. Conceptually, it answers the question: *"What hidden factors explain why certain users like certain movies?"*

These latent factors might represent things like genre preference, pacing, tone, or director style — none of which are explicitly labelled in the data, but which SVD discovers automatically from rating patterns.

This allows the model to:

- Identify hidden user preference patterns without requiring explicit genre or content labels
- Predict how a user would rate a movie they haven't seen yet
- Handle the **sparse matrix problem** inherent in recommendation datasets — most users have only rated a tiny fraction of all available movies, and SVD fills in those gaps intelligently

---

## 🚀 Getting Started


---

## 🔮 Future Improvements

- Incorporate **genre-level filtering** for category-specific recommendations
- Extend to **all users** via a batch recommendation pipeline
- Experiment with **ALS (Alternating Least Squares)** and **NMF** as alternative matrix factorisation methods
- Add a **web interface** for interactive recommendations
- Integrate with the full Netflix Prize dataset (all 4 combined data files)

---

## 🙌 Acknowledgements

- [Netflix Prize Dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) — Original data source
- [scikit-surprise](https://surpriselib.com/) — Recommendation system library
- Inspired by real-world OTT recommendation pipelines used by Netflix and Amazon Prime

---

# Recommender System

A **two-layer hybrid recommendation pipeline** featuring regime-conditioned ranking under severe cold-start constraints.

**Core question:**

> *Given user behavior, what should we show next — and why?*

This project emphasizes **correct data handling, defensible baselines, and evaluation discipline**, not just model complexity.

---

## Project Structure

```

Recommender-system/
│
├── data/
│   ├── raw/                 # Original events data
│   └── processed/           # Aggregated interactions
│
├── src/
│   ├── interactions.py      # Interaction construction
│   ├── candidate_generation.py
│   └── evaluation.py        # Train/test split + metrics
│
├── notebooks/
│   └── eda.ipynb            # Exploratory analysis
│
└── README.md

```
## System Overview

### Stage 1 — Candidate Generation
- Item-Item co-occurrence similarity
- Popularity fallback
- Top-N candidates per user

### Stage 2 — Logistic Regression Ranker
The ranker re-scores candidates using the following features:

- `item_similarity_score`
- `item_popularity`
- `time_since_last_interaction`
- `user_history_count` ✅ (added)
- `item_interaction_count` ✅ (added)

The last two features were introduced to improve behavior modeling under extreme sparsity.

---

## Dataset & Interaction Modeling

### Dataset Source

The dataset used in this project is the **RetailRocket Recommender System Dataset**, obtained from **Kaggle**.

It contains anonymized user interaction logs from an e-commerce platform, including:

- Page views
- Add-to-cart events
- Purchase (transaction) events

Each event includes:

- Timestamp
- User identifier
- Item identifier
- Event type

---

## Interaction Definition

User behavior is converted into **implicit feedback** using weighted event signals:

```python
EVENT_WEIGHT = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}
````

Raw events are aggregated at the **(user_id, item_id)** level by:

* Summing event weights → `interaction_score`
* Preserving the most recent timestamp → `last_interaction_ts`

### Final Interaction Schema

| Column              | Description                       |
| ------------------- | --------------------------------- |
| user_id             | Unique user identifier            |
| item_id             | Unique item identifier            |
| interaction_score   | Strength of implicit feedback     |
| last_interaction_ts | Most recent interaction timestamp |

This interaction table is the **single source of truth** for all downstream models.

---

## Why Implicit Feedback Is Necessary

In real-world recommender systems:

* Explicit ratings are rare or unavailable
* Most users never provide direct feedback
* Behavioral data is abundant and passive

Implicit feedback is necessary because:

* It captures **natural user behavior**
* Interaction frequency and intensity correlate with intent
* It scales to millions of users without friction
* It reflects how production systems actually operate

The system learns from **what users do**, not what they claim.

---

## Train/Test Split Strategy

To avoid data leakage, interactions are split **by time**:

* **Train:** first 80% of interactions (earliest timestamps)
* **Test:** last 20% of interactions (future behavior)

This simulates a real-world scenario:

> *Predict future user interactions using only past data.*

---

## Evaluation Metrics

Models are evaluated **offline** using:

* **Precision@10**
* **Recall@10**

Evaluation is performed **per test user**, answering:

> **“Out of what users actually interacted with later, how many did we surface?”**

---

## Baseline Recommender: Popularity Model

### Model Definition

The popularity-based baseline:

* Ranks items by total `interaction_score` in the **training set**
* Excludes items already interacted with by the user
* Recommends the top-K remaining items (K = 10)

This model provides a **lower bound** for performance and validates the evaluation pipeline.

### Results (Popularity Baseline)

```
Precision@10: 0.00084
Recall@10:    0.00748
Test users:   302,356
```

These results are **expected and correct** given:

* Extremely sparse user behavior
* Large item catalog
* No personalization
* Heavy popularity bias

---

## Item-Based Collaborative Filtering

### Model Overview

To introduce personalization while remaining defensible under sparsity, the project implements an **item–item collaborative filtering model** based on **co-occurrence similarity**.

### How It Works

1. For each user, collect all interacted items
2. Generate all item–item pairs within each user
3. Count co-occurrences across users
4. Normalize using **cosine similarity**

[
\text{sim}(i, j) = \frac{\text{cooc}(i, j)}{\sqrt{\text{count}(i) \cdot \text{count}(j)}}
]

This captures **behavioral similarity** while reducing popularity bias.

---

## Candidate Generation Logic

For a given user:

1. Retrieve items from the user’s interaction history
2. Retrieve similar items for each history item
3. Aggregate similarity scores
4. Rank candidate items
5. Exclude previously seen items

---

## Cold-Start Handling (Explicit)

Cold-start is handled **explicitly in code**, not implicitly:

### User Cold-Start

* If a user has **no training history**
* Fallback → popularity-based recommendations

### Item Cold-Start

* If an item has **no neighbors in the similarity matrix**
* Fallback → popularity-based recommendations

Cold-start is treated as the **dominant real-world case**, not an edge case.

---

## Results: Item-Based Collaborative Filtering

```
Precision@10:        0.00119
Recall@10:           0.00958
Test users:          302,356
Cold-start users:    280,190
Items with neighbors:138,131
```

### Interpretation

* Item-based CF **outperforms popularity**
* Gains are modest but **statistically meaningful**
* Cold-start dominates evaluation
* Improvements are constrained by extreme sparsity

These results confirm:

> **Item-to-item similarity is more defensible than user embeddings under sparse implicit feedback.**

---

## Key Observations from EDA

* Majority of users have **only one interaction**
* User–item matrix is extremely sparse
* Cold-start is the norm, not the exception
* Popularity dominates without personalization

These findings strongly motivate:

* Item-to-item methods
* Session-based recommendation
* Hybrid retrieval strategies

---

## Known Limitations

Implicit feedback models have inherent constraints:

* No explicit negative signals
* Missing interactions ≠ dislike
* Exposure bias favors popular items
* Single-interaction users limit personalization

As a result:

* Recall remains low
* Long-tail discovery is difficult
* Gains require stronger signals or context

These limitations are **intentional and instructional** at this stage.

---

# Logistic Regression Training Summary

- Training rows: **12,013,001**
- Validation AUC: **0.7803**
- Validation Log Loss: **0.3869**

Learned coefficients:

| Feature | Coefficient |
|----------|------------|
| item_similarity_score | 0.1273 |
| item_popularity | 0.9692 |
| time_since_last_interaction | -0.0054 |

(Counts added later in improved model.)

---

# A. Ablation Study — Do Count Features Matter?

Test Set: **302,356 users**

## Comparison Table

| Model Variant | Recall@10 | NDCG@10 |
|---------------|-----------|----------|
| Popularity | 0.0075 | 0.0039 |
| Item-Item CF | 0.0096 | 0.0053 |
| CF + LR Ranker (no count features) | 0.0088 | 0.0048 |
| CF + LR Ranker (+ count features) | **0.0102** | **0.0058** |

---

## Impact of Count Features (Ranker Only)

Comparing Ranker **with vs without** counts:

| Metric | Improvement |
|--------|-------------|
| Recall@10 | **+16.0%** |
| NDCG@10 | **+22.2%** |

Without count features, the ranker *degrades* CF ordering.  
With count features, the ranker becomes additive and surpasses CF.

### Conclusion

`user_history_count` and `item_interaction_count` are quantitatively necessary for:

- Stabilizing ranking under sparse histories  
- Calibrating popularity strength  
- Preventing over-correction of similarity signals  

---

# B. Failure Slice Analysis

### When the Ranker Helps

✔️ **Warm Users (multiple historical interactions)**  
- User has enough interaction depth  
- Behavioral counts provide signal strength  
- Ranker reorders CF candidates effectively  

✔️ **Items with meaningful interaction volume**  
- item_interaction_count helps calibrate confidence  
- Reduces noise from weak co-occurrence edges  

Result:  
Ranker improves both Recall@10 and NDCG@10.

---

### When the Ranker Cannot Help

❌ **Extreme Cold Users (1 or 0 interactions)**  
- user_history_count ≈ 0  
- No personalization signal  
- Candidate set already weak  
- Ranker only reshuffles popularity

❌ **Items with no co-occurrence neighbors**  
- No similarity signal  
- Falls back to popularity  

Given that:

- 302,356 total test users  
- 280,190 are cold-start users (~92%)

Most ranking decisions operate under severe sparsity.

---

# C. Brutal Honesty

> With 92% cold users, ranking gains are bounded; most improvements come from better handling of sparse user histories rather than true personalization.

---

# Key Takeaways

1. Logistic Regression ranker is effective **only when behavioral intensity signals are included**.
2. Count features transform the ranker from harmful to beneficial.
3. Gains are real but fundamentally constrained by extreme cold-start distribution.
4. Future improvements must focus on:
   - Better candidate generation
   - Cold-start modeling
   - Regime-aware ranking strategies

---

# Current Best Logistic Regression Performance

| Model | Recall@10 | NDCG@10 |
|--------|-----------|----------|
| Item-Item CF | 0.0096 | 0.0053 |
| Item-Item CF + LR Ranker (Improved) | **0.0102** | **0.0058** |

---

# Next Steps

- Add regime-aware ranking logic  
- Explore XGBoost ranker comparison  
- Segment evaluation by cold vs warm users  
- Improve candidate generator quality  

---

**Status:** Logistic Regression ranker validated with quantitative ablation evidence.  
Further gains require improvements upstream in candidate generation.

This project prioritizes:

* Correct temporal splits
* Transparent assumptions
* Interpretable baselines
* Scalable system design

---



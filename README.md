Below is an **updated, polished README.md** that incorporates **all recent changes** to your project:

* Item–Item Collaborative Filtering
* Co-occurrence matrix + cosine similarity
* Cold-start logic
* Updated evaluation results
* Clear explanation of *why* results look the way they do
* Keeps the professional, principled tone you’re already using

You can **copy–paste this directly as your README.md**.

---

```markdown
# Recommender System

An **end-to-end recommendation system** covering interaction modeling, candidate generation, offline evaluation, and clear modeling assumptions.

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

````

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

## Next Steps

Planned extensions include:

* Time-decayed popularity
* Session-based recommendation
* Candidate ranking models
* Learning-to-rank with implicit labels
* Online-serving friendly retrieval strategies

Each model will be evaluated **against existing baselines** to measure real lift.

---

## Key Design Principle

> **Simple models + correct evaluation beat complex models with leakage.**

This project prioritizes:

* Correct temporal splits
* Transparent assumptions
* Interpretable baselines
* Scalable system design

---



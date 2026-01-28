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
```

---

## Dataset & Interaction Modeling

### Dataset Source

The dataset used in this project is the **RetailRocket Recommender System Dataset**, obtained from **Kaggle**.

It contains anonymized user interaction logs from an e-commerce platform, including:

* Page views
* Add-to-cart events
* Purchase (transaction) events

Each event includes:

* Timestamp
* User identifier
* Item identifier
* Event type

---

## Interaction Definition

User behavior is converted into **implicit feedback** using weighted event signals:

```python
EVENT_WEIGHT = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}
```

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

## Baseline Recommender: Popularity Model

### Model Definition

The first recommender implemented is a **popularity-based baseline**:

* Items are ranked by total `interaction_score` in the **training set**
* Items already interacted with by the user are excluded
* The top-K remaining items are recommended (K = 10)

This model provides a **lower bound** for performance and validates the pipeline.

---

## Train/Test Split Strategy

To avoid data leakage, interactions are split **by time**:

* **Train:** first 80% of interactions (earliest timestamps)
* **Test:** last 20% of interactions (future behavior)

This simulates a real-world scenario:

> *Predict future user interactions using only past data.*

---

## Evaluation Metrics

The recommender is evaluated **offline** using:

* **Precision@10**
* **Recall@10**

Evaluation is performed **per test user**, answering:

> **“Out of what users actually interacted with later, how many did we surface?”**

### Example Results (Popularity Baseline)

```python
{
  'Precision@10': 0.00084,
  'Recall@10': 0.00748,
  'num_test_users': 302,356
}
```

These results are **expected and correct** given:

* Extremely sparse user behavior
* Large item catalog
* Lack of personalization

---

## Key Observations from EDA

* A majority of users have **only one interaction**
* User-item interaction matrix is highly sparse
* Popularity dominates without personalization
* Cold-start behavior is the norm, not the exception

These observations strongly motivate **item-to-item** and **session-based** approaches.

---

## Known Limitations

Implicit feedback models have inherent constraints:

* No explicit negative signals
* Missing interactions ≠ dislike
* Exposure bias favors popular items
* Single-interaction users limit personalization

As a result, popularity-based recommenders:

* Perform poorly for recall
* Lack personalization
* Over-recommend head items

These limitations are **intentional** at this stage — the baseline exists to be beaten.

---

## Next Steps

Planned extensions include:

* Item-to-item similarity (co-occurrence-based)
* Time-decayed popularity
* Session-based recommendation
* Candidate ranking models
* Online-serving friendly retrieval strategies

Each model will be evaluated **against this baseline** to measure real lift.

---

## Key Design Principle

> **Simple models + correct evaluation beat complex models with leakage.**

This project prioritizes:

* Correct data splits
* Transparent assumptions
* Interpretable baselines
* Scalable system design

---


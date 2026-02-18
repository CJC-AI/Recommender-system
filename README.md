# Recommender System — RetailRocket Dataset

A **two-stage hybrid recommendation system** built under extreme sparsity and cold-start constraints.

**Core question:**

> Given past user behavior, what should we show next — and why?

This project focuses on system design, correct evaluation, and bottleneck diagnosis — not just model complexity.

---

# 1. Problem Context

The dataset presents a difficult real-world scenario:

* **302,356 test users**
* **~92% are cold-start users** (only one interaction)
* **89% of items are never purchased**
* Extremely sparse user–item matrix

This means:

* Personalization is heavily constrained
* Candidate generation becomes the primary bottleneck
* Ranking alone cannot solve the problem

This project is designed to measure and diagnose those limits clearly.

---

# 2. Project Structure

```
Recommender-system/
│
├── data/
│   ├── raw/
│   │   ├── events.csv
│   │   ├── category_tree.csv
│   │   ├── item_properties_part1.csv
│   │   └── item_properties_part2.csv
│   │
│   └── processed/
│       ├── interactions.csv
│       └── training_data.csv
│
├── src/
│   ├── interactions.py
│   ├── candidate_generation.py
│   ├── ranking/
│   │   ├── dataset.py
│   │   ├── train_lr.py
│   │   ├── train_xgb.py
│   │   ├── infer.py
│   │   └── evaluation.py
│
├── notebooks/
│   └── eda.ipynb
│
└── README.md
```

---

# 3. System Architecture

The system follows a **retrieval → ranking pipeline**, similar to production recommender systems.

---

## Stage 1 — Candidate Generation (Retrieval Layer)

Candidates are generated using:

1. **Item–Item Collaborative Filtering**

   * Cosine similarity over co-occurrence counts
   * Normalized to reduce popularity bias

2. **Popularity Fallback**

   * Used for cold-start users or isolated items

3. **Taxonomy Augmentation**

   * Uses `category_tree.csv`
   * Fills candidate gaps using category neighbors

4. **Configurable Retrieval Width**

   * Top 50 candidates
   * Extended to Top 100 after bottleneck diagnosis

---

## Stage 2 — Learning-to-Rank

A Logistic Regression model re-scores candidates using:

* `item_similarity_score`
* `item_popularity`
* `time_since_last_interaction`
* `user_history_count` ✅
* `item_interaction_count` ✅

The last two features allow the model to:

* Detect cold vs warm users
* Adjust behavior dynamically
* Avoid over-correcting similarity signals

Interaction weights are used only for **sample weighting**, not as features (to avoid leakage).

---

# 4. Interaction Modeling

Implicit feedback is constructed using weighted event signals:

```python
EVENT_WEIGHT = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}
```

Events are aggregated at the `(user_id, item_id)` level:

* `interaction_score` (sum of weights)
* `last_interaction_ts` (most recent timestamp)

This forms the single source of truth for training.

---

# 5. Train/Test Split

Data is split **by time**:

* First 80% → Train
* Last 20% → Test

This ensures:

* No future leakage
* Realistic next-item prediction

---

# 6. Evaluation Metrics

Evaluated per user using:

* **Recall@10**
* **NDCG@10**

Recall answers:

> Did we include the correct item?

NDCG answers:

> Did we rank it high?

---

# 7. Baseline Performance

## Popularity Model

| Metric    | Value  |
| --------- | ------ |
| Recall@10 | 0.0075 |
| NDCG@10   | 0.0039 |

---

## Item–Item CF

| Metric    | Value  |
| --------- | ------ |
| Recall@10 | 0.0096 |
| NDCG@10   | 0.0053 |

Modest improvement, but heavily constrained by cold-start dominance.

---

# 8. Ranker Ablation Study

| Model Variant               | Recall@10 | NDCG@10 |
| --------------------------- | --------- | ------- |
| Popularity                  | 0.0075    | 0.0039  |
| Item-Item CF                | 0.0096    | 0.0053  |
| CF + LR (no count features) | 0.0088    | 0.0048  |
| CF + LR (+ count features)  | 0.0102    | 0.0058  |

### Impact of Count Features

| Metric    | Improvement |
| --------- | ----------- |
| Recall@10 | +16.0%      |
| NDCG@10   | +22.2%      |

Without count features, the ranker harms performance.
With count features, the ranker becomes additive.

---

# 9. System Bottleneck Diagnosis

This project explicitly measures candidate reachability.

## Candidate Recall@50 (Behavior Only)

```
0.0240
```

This means:

> The correct item is not even visible 97.6% of the time.

Ranking cannot recover what retrieval does not surface.

---

## After Taxonomy Augmentation

Candidate Recall@50:

```
0.0249
```

Small improvement (~3.75%).

Taxonomy improves semantic coverage more than exact item ID recovery.

---

## After Increasing Retrieval Width to 100

Candidate Recall@100:

```
0.0362
```

This raises the theoretical Recall@10 ceiling by ~50%.

This confirms:

> Retrieval width was the dominant bottleneck.

---

# 10. Final Performance (After Retrieval Expansion)

| Model                    | Recall@10  | NDCG@10    |
| ------------------------ | ---------- | ---------- |
| Popularity               | 0.0075     | 0.0039     |
| Item-Item CF             | 0.0096     | 0.0053     |
| CF + LR Ranker (Top 100) | **0.0111** | **0.0062** |

Improvement achieved **without modifying the ranker** — purely by expanding retrieval.

This confirms that:

> Candidate generation was the primary system constraint.

---

# 11. Cold-Start Reality

* 302,356 test users
* 280,190 cold-start users (~92%)

Most ranking decisions occur under minimal personalization signal.

Gains are therefore bounded by data structure, not model complexity.

---

# 12. Known Limitations

* No exposure logs (negatives are sampled)
* No session-based modeling
* No sequential dynamics
* No personalization for single-interaction users
* Exact-item matching evaluation penalizes category-level relevance

These limitations are intentional and documented.

---

# 13. Key Engineering Takeaways

1. Retrieval bottlenecks matter more than ranking sophistication.
2. Count features stabilize ranking under sparse regimes.
3. Increasing candidate width can outperform adding model complexity.
4. Proper leakage control is essential.
5. Most improvements in sparse systems come from handling cold-start correctly.

---

# 14. Future Improvements

* Session-based next-item modeling
* Time-decayed co-visitation weighting
* Bayesian smoothing of item counts
* Personalized popularity priors
* Exposure-aware negative sampling

---

# 15. Project Philosophy

This project prioritizes:

* Clear system decomposition
* Honest bottleneck measurement
* Transparent assumptions
* Interpretable baselines
* Realistic evaluation
* Practical scalability

The goal is not to maximize metrics blindly.

The goal is to understand **why the system behaves the way it does.**

---

**Status:**
Two-stage hybrid recommender system validated with retrieval expansion, ablation evidence, and bottleneck diagnosis under extreme sparsity.

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
│   ├── taxonomy.py               # Category-aware backfill engine
│   ├── analysis/
│   │   └── failure_analysis.py   # Diagnostics for model disagreement
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
   * Fills candidate gaps using category neighbors when CF is sparse

4. **Configurable Retrieval Width**

   * Top 50 candidates
   * Extended to Top 100 after bottleneck diagnosis

---

## Stage 2 — Learning-to-Rank

Multiple rankers were trained (Logistic Regression vs. XGBoost). The final deployed model uses **Logistic Regression** based on failure analysis results.

**Features Engineered:**
* `item_similarity_score` (Behavioral signal)
* `user_category_affinity` (Semantic signal)
* `item_interaction_count` (Global confidence signal)
* `user_history_count` (User maturity signal)
* `time_since_last_interaction` (Temporal decay)

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

| Metric | Value |
| --- | --- |
| Recall@10 | 0.0075 |
| NDCG@10 | 0.0039 |

---

## Item–Item CF

| Metric | Value |
| --- | --- |
| Recall@10 | 0.0096 |
| NDCG@10 | 0.0053 |

Modest improvement, but heavily constrained by cold-start dominance.

---

# 8. Ranker Ablation Study

| Model Variant | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Popularity | 0.0075 | 0.0039 |
| Item-Item CF | 0.0096 | 0.0053 |
| CF + LR (no count features) | 0.0088 | 0.0048 |
| CF + LR (+ count features) | 0.0102 | 0.0058 |

### Impact of Count Features

| Metric | Improvement |
| --- | --- |
| Recall@10 | +16.0% |
| NDCG@10 | +22.2% |

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

# 10. Final Performance Comparison

We compared a linear baseline (Logistic Regression) against a non-linear tree ensemble (XGBoost).

| Model | Recall@10 | NDCG@10 |
| --- | --- | --- |
| Popularity | 0.0075 | 0.0039 |
| Item-Item CF | 0.0096 | 0.0053 |
| CF + XGBoost (Top 100) | 0.0056 | 0.0034 |
| **CF + LR Ranker (Top 100)** | **0.0113** | **0.0063** |

**Winner:** Logistic Regression.

---

# 11. Failure Analysis: The "AUC Illusion"

Why did XGBoost fail despite having higher AUC (0.87) compared to LR (0.80)?

### Diagnosis

We ran a `failure_analysis.py` script to inspect disagreements between the models.

1. **The Safety Trap:**
* XGBoost learned that low `item_interaction_count` strongly predicts a "non-click."
* It systematically suppressed "niche" items (high similarity, low volume) in favor of safe, moderately popular items.
* In a ranking task, finding the specific niche item is the goal; suppressing it maximizes AUC but kills Recall.


2. **Shortcut Learning:**
* XGBoost over-indexed on `user_category_affinity`.
* It effectively learned: "If the category matches, score it high."
* This ignored the finer-grained `item_similarity_score`, causing it to recommend *any* item from the right category rather than the *specific* item the user viewed.


3. **Linearity as a Feature:**
* Logistic Regression is additive: `Score = (0.25 * Similarity) + (0.90 * Popularity)`.
* A very strong Similarity signal can mathematically override a weak Popularity signal.
* This property is crucial for sparse/long-tail recommendation, where the best item often has low global popularity.



---

# 12. Cold-Start Reality

* 302,356 test users
* 280,190 cold-start users (~92%)

Most ranking decisions occur under minimal personalization signal.

Gains are therefore bounded by data structure, not model complexity.

---

# 13. Known Limitations

* No exposure logs (negatives are sampled)
* No session-based modeling
* No sequential dynamics
* No personalization for single-interaction users
* Exact-item matching evaluation penalizes category-level relevance

These limitations are intentional and documented.

---

# 14. Key Engineering Takeaways

1. **High AUC  Good Ranking.** A model can be excellent at classifying negatives (AUC) while being terrible at sorting positives (NDCG).
2. **Linear models often beat trees in sparse regimes.** Their additive nature forces them to respect weak signals (like similarity) that trees might gate out as "noise."
3. **Retrieval bottlenecks dominate.** Increasing candidate width from 50 to 100 provided a larger gain than any modeling change.
4. **Feature Engineering > Model Complexity.** Adding `user_category_affinity` and `counts` mattered more than switching to XGBoost.

---

# 15. Future Improvements

* Session-based next-item modeling (RNN/Transformer) to capture sequence.
* Time-decayed co-visitation weighting.
* Bayesian smoothing of item counts.
* Personalized popularity priors.
* Exposure-aware negative sampling.

---

# 16. Project Philosophy

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
Two-stage hybrid recommender system validated. **Logistic Regression selected for deployment** after diagnostics proved XGBoost over-optimized for safety over relevance.

```

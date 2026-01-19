# Recommender-system
End-to-End Recommendation System with Ranking, Evaluation, and Serving. Given user behavior, what should we show next and why? 

# Dataset & Interaction Modeling

## Dataset Source
The dataset used in this project is the **RetailRocket Recommender System Dataset**, obtained from **Kaggle**.  
It contains anonymized user interaction logs from an e-commerce website, including:

- Page views
- Add-to-cart events
- Purchase (transaction) events

Each event includes a timestamp, user identifier, item identifier, and event type.

---

## Interaction Definition
User behavior is converted into **implicit feedback signals** using weighted interactions:

```python
EVENT_WEIGHT = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0
}
````

Interactions are aggregated at the **(user_id, item_id)** level by:

* Summing event weights to produce an `interaction_score`
* Preserving the most recent interaction timestamp

Final interaction schema:

| Column              | Description                         |
| ------------------- | ----------------------------------- |
| user_id             | Unique user identifier              |
| item_id             | Unique item identifier              |
| interaction_score   | Weighted implicit feedback strength |
| last_interaction_ts | Most recent interaction timestamp   |

---

## Why Implicit Feedback Is Necessary

In real-world e-commerce systems, **explicit feedback (ratings, reviews)** is often unavailable or sparse.

Implicit feedback is necessary because:

* Most users do not leave ratings
* Behavioral signals (views, cart actions, purchases) occur naturally
* Interaction frequency and intensity correlate with user intent
* It enables scalable recommendation systems without user friction

Implicit signals allow the model to infer preferences from **what users do**, not what they say.

---

## Known Limitation

A key limitation of implicit feedback modeling is the **lack of negative signals**:

* A missing interaction does not mean dislike
* Viewing an item does not guarantee interest
* Popular items may dominate recommendations due to exposure bias

As a result, the model may:

* Over-recommend popular items
* Struggle to distinguish curiosity from genuine intent

Mitigation strategies include time decay, confidence weighting, and hybrid recommendation approaches.
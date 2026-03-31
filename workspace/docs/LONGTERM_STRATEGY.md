# Long-term strategy (two tracks)

## 1. Feature selection and training domain

**Goal:** Robusted training data should sit as close as practical to **the validator’s domain**: same kinds of hands, miner-visible sanitization, chunking, and **the same feature extraction** as at inference.

- If you build, filter, or augment training so its feature distributions align with what you see in **logged validator requests** (or you mix log-derived rows into train), you are deliberately **pulling training toward deployment**.
- You will **not** match live traffic perfectly (sampling noise, bot mix changes, policy updates); a residual gap is normal.

**Takeaway:** Yes — it is possible and sensible to aim feature selection and the main supervised model at the validator domain, as long as the pipeline matches production and logs are used as **distribution targets**, not as a substitute for ground-truth labels.

---

## 2. After training: SSL and KNN on logs

**SSL (on logs):** Learn a representation from **unlabeled** traffic so the model sees plenty of validator-style data without needing human/bot labels on those rows.

**KNN (unsupervised, on a growing log):** After encoding batches or chunks, use **nearest neighbors in embedding space** as a **monitoring** layer: similarity to past traffic, drift, or rough anomaly detection. You need **enough stored embeddings** (or consistent features) over time and the **same encoder** you use at runtime.

**Caveat:** KNN answers *“does this look like my history?”* — not *“what is the validator’s true label?”* Keep the **supervised head** (trained on labeled data) as the source of bot risk for scoring; use SSL + KNN to support robustness and operational awareness, not as a drop-in replacement for labeled training.

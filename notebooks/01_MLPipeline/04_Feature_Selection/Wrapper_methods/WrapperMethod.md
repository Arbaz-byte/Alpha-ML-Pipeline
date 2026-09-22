# Wrapper Methods for Feature Selection — Reference Guide


## 1. Introduction — What Is a Wrapper Method?

* **Filter** methods score each feature by a statistic computed independent of any model
* **Embedded** methods get feature importance as a side effect of training one regularized model

* **wrapper** methods sit in between — they treat feature selection as a search problem, where the "fitness" of a candidate feature subset is measured by **actually training a model on it and scoring the result**.

given a full feature set $F = \{f_1, f_2, \dots, f_n\}$, a wrapper method searches for the subset $S \subseteq F$ that maximizes a model's cross-validated performance:

$$
S^* = \underset{S \subseteq F,\ S \neq \emptyset}{\arg\max} \ \text{CV}\_\text{Score}(\mathcal{M}(S))
$$

where $\mathcal{M}(S)$ denotes the model trained using only the features in $S$, and `CV_Score` is a cross-validated metric ($R^2$, accuracy, ROC-AUC, etc.).

The search space is the **power set** of $F$ — $2^n - 1$ non-empty candidate subsets — which is why every wrapper method is really a different strategy for searching that space without paying the full $2^n$ cost every time.

**What makes this family different from filters:** the score a wrapper method optimizes is *exactly* the metric that will matter in production, evaluated on the *exact* model that will be deployed. That's strictly more informative than any univariate statistic and strictly more expensive, since it means training a model over and over rather than computing a formula once per feature.

---

## 2. The Four Methods Covered

### 2.1 Exhaustive Feature Selection (EFS)

**What it does:** evaluates literally every non-empty subset of $F$ and keeps the one with the best cross-validated score. This is the direct, brute-force solution to the formal objective above — no approximation.

$$
S^*_{\text{EFS}} = \underset{S \in \mathcal{P}(F) \setminus \{\emptyset\}}{\arg\max} \ \text{CV}\_\text{Score}(S), \qquad |\mathcal{P}(F) \setminus \{\emptyset\}| = 2^n - 1
$$

**Time complexity:** $O(2^n \cdot T_{\text{fit}})$, where $T_{\text{fit}}$ is the cost of one cross-validated model fit. Guaranteed optimal; only tractable for small $n$ (roughly $n \leq 15$–$20$ in practice).

### 2.2 Sequential Forward Selection (SFS)

**What it does:** starts from an empty set and greedily adds, one at a time, whichever remaining feature improves the score the most.

$$
S_0 = \emptyset, \qquad S_{k+1} = S_k \cup \{ \underset{f \in F \setminus S_k}{\arg\max} \ \text{CV}\_\text{Score}(S_k \cup \{f\}) \}
$$

Run to completion, this produces a full path from $|S|=1$ to $|S|=n$, and the best point along that path (or the smallest point within one standard error of the best — see Section 5) is the final answer.

**Time complexity:** $O(n^2 \cdot T_{\text{fit}})$ — round $k$ evaluates $n-k+1$ candidates, so total evaluations sum to $\frac{n(n+1)}{2}$. Greedy and irreversible: once a feature is added, it's never reconsidered.

### 2.3 Backward Elimination

**What it does:** the mirror image of forward selection — start with every feature, and at each step remove whichever single feature costs the least when dropped.

$$
S_0 = F, \qquad S_{k+1} = S_k \setminus \{ \underset{f \in S_k}{\arg\max} \ \text{CV}\_\text{Score}(S_k \setminus \{f\}) \}
$$

**Time complexity:** identical to forward, $O(n^2 \cdot T_{\text{fit}})$, for the same summed-round reason. The key structural difference from forward is *when* it commits to a decision: backward always decides while seeing the full remaining feature context; forward decides while seeing only what it's already built up.

### 2.4 Recursive Feature Elimination (RFE)

**What it does:** the cheapest of the four. Fit the model once on the current feature set, read off the model's **own internal importance ranking** (coefficient magnitude for linear models, `feature_importances_` for trees), and drop the single worst-ranked feature — no re-scoring of alternatives.

$$
c_i = |w_i| \ \text{(linear models)} \quad \text{or} \quad c_i = \text{importance}_i \ \text{(tree-based models)}
$$

$$
S_{k+1} = S_k \setminus \{ \underset{i \in S_k}{\arg\min} \ c_i \}
$$

**Time complexity:** $O(n \cdot T_{\text{fit}})$ — one fit per round, $n$ rounds total, no inner loop over candidates. This is the structural reason it's cheaper than forward/backward: it never tests whether removing a *different* feature would have scored better, it trusts the model's own ranking instead.


---

## 3. Notebooks Built

| # | Notebook | Method | Dataset(s) / Task | Key Finding |
|---|---|---|---|---|
| 1 | `01_Exhaustive_Feature_Selection.ipynb` | EFS | Iris (classification) + Concrete (regression) | Ground-truth baseline for the series — every subsequent method is measured against what EFS found here |
| 2 | `02_Forward_Feature_selection.ipynb` | SFS (forward) | Boston Housing (regression) | Found the exact global optimum (11 features, $R^2=0.7102$) matching EFS with 90x fewer evaluations (91 vs. 8,191); one-SE rule selected 6 features that *overfit less* than the 13-feature baseline |
| 3 | `03_Backward_Feature_Elimination.ipynb` | SFS (backward) | Boston Housing (regression) | Same 6-feature one-SE answer as forward, but a different path — found a better 5-feature subset than forward did (the exact gap forward missed against EFS), evidence that starting from full context avoids some greedy mistakes |
| 4 | `04_Recursive_Feature_Elimination.ipynb` | RFE (+ 4-way comparison) | Boston Housing (regression) + binarized-target classification (ROC-AUC) | Matched the global optimum but lagged forward/backward at intermediate sizes (mean gap 0.0084 vs. 0.0010/0.0003); that noisier path pushed RFE's *own* one-SE rule to select 10 features instead of 6 — a real, measured trade-off, not a theoretical one |

---

## 4. Comparison Table — All Four Methods

| | EFS | Forward | Backward | RFE |
|---|---|---|---|---|
| Starts from | — (tries everything) | Empty set | Full set | Full set |
| Fits for $n=13$ | 8,191 | 91 | 91 | 13 |
| Time complexity | $O(2^n)$ | $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| Optimality guarantee | Yes | No | No | No |
| Decides removal/addition by | Trying every subset | Re-scoring every candidate | Re-scoring every candidate | Model's own importance ranking |
| Requires `coef_`/`feature_importances_` | No | No | No | **Yes** |
| Works with any model | Yes | Yes | Yes | No |
| Global optimum found (Housing, 11 features) | Yes (by definition) | Yes | Yes | Yes |
| One-SE final subset size (Housing) | 6 | 6 | 6 | **10** |
| Mean CV-score gap to true optimum, across sizes | 0 (is the optimum) | 0.0010 | 0.0003 | 0.0084 |
| Best suited for | Very small $n$ (≲ 15–20) | Moderate $n$, any model | Moderate $n$, full-model fit is feasible | Large $n$, linear/tree model available |
| Practically infeasible when | $n$ large | — | $n > $ number of samples, or model can't fit on all features at once | Model has no importance/coefficient attribute |

---

## 5. Final Verdict

Wrapper methods are the **model-aware** layer of feature selection — they answer "does removing this feature change what my actual model gets right," which no filter method can answer and which embedded methods only answer indirectly (as a side effect of one regularization run). Across this series, that model-awareness paid off in a concrete way every notebook: Adjusted R² consistently favored the wrapper-selected subset over the full-feature baseline, and in two notebooks (Forward, RFE) the smaller model measurably **overfit less**, not just scored the same with fewer inputs.

But the cost is real and scales badly with the naive version of the idea. Exhaustive search is only a reference method past a small feature count — its entire value in this series was as the ground truth the other three got checked against, not as something to reach for on a real dataset with more than a handful of features. Forward and backward bring that down to a tractable $O(n^2)$, and this series showed they don't just theoretically differ — backward found a genuinely better intermediate subset than forward did on identical data, a direct, measured consequence of deciding what to drop with full context versus building up blind. RFE brings the cost down further still, to $O(n)$, and that saving is not free: its own results here showed both a real accuracy lag *and* a downstream effect on the final selected subset size, once the same stopping rule was applied to its noisier path.

**The practical takeaway:** there is no single best wrapper method — there's a genuine cost/quality dial, and where to set it depends on $n$, the model in use, and whether a `coef_`/`feature_importances_` attribute is even available. On wide data (hundreds of features), no wrapper method here is the right first move at all — a filter pass (Variance Threshold, correlation) should cut dimensionality first, and a wrapper method should only run on the survivors. That's the same conclusion the Filter Methods reference reached from the other direction, and it's the throughline of both series: no single feature selection technique is complete on its own, and a real pipeline chains several of them, checking each one's output against an actual trained model rather than trusting any single statistic or search in isolation.

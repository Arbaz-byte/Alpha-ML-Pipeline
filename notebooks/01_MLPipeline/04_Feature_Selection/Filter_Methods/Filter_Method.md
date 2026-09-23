# Filter Methods for Feature Selection

## 1. What Is a Filter Method?

Feature selection techniques generally fall into three families:

| Family | How it decides | Model involved? | Cost |
|---|---|---|---|
| **Filter** | Statistical property of the feature itself (vs. target or vs. other features) | No — model-agnostic | Cheap, fast |
| **Wrapper** | Trains a model repeatedly on feature subsets, keeps the best-performing subset | Yes — search-driven | Expensive |
| **Embedded** | Feature importance falls out of training a single model (e.g. Lasso coefficients, tree feature importances) | Yes — built into training | Moderate |

**Filter methods** score each feature (or feature pair) using a statistical test — entropy, a chi-square statistic, variance, correlation, an F-statistic — *before* any model is trained. They don't know or care what model comes next; they only ask "does this feature carry information, on its own terms?" That makes them the fastest, simplest, first-pass layer in a feature selection pipeline — usually run before a wrapper or embedded method narrows things further.

---

## 2. The Five Methods Covered

### 2.1 Information Gain

**What it measures:** the reduction in entropy (uncertainty about the target) achieved by knowing a feature's value. Rooted in information theory — the same math behind decision tree splits.

$$
\text{Entropy}(S) = -\sum_i p_i \log_2(p_i)
$$

$$
\begin{aligned}
\text{InformationGain}(S, A) &= \text{Entropy}(S) \\
&\quad - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
\end{aligned}
$$

Higher Information Gain = the feature does more to sort the target's classes apart.

## 2. The Five Methods Covered

### 2.1 Information Gain

**What it measures:** the reduction in entropy (uncertainty about the target) achieved by knowing a feature's value. Rooted in information theory — the same math behind decision tree splits.

$$
\text{Entropy}(S) = -\sum_i p_i \log_2(p_i)
$$

$$
\text{InformationGain}(S, A)
= \text{Entropy}(S)
- \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
$$

Higher Information Gain = the feature does more to sort the target's classes apart.

### 2.2 Chi-Square ($\chi^2$) Test

**What it measures:** independence between two **categorical** variables, via a contingency table of observed vs. expected counts.

$$
\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}
$$

where $O_i$ is the observed count and $E_i$ is the expected count.

$H_0$: feature and target are independent. A large $\chi^2$ (→ small p-value) means the observed counts deviate from what independence would predict — reject $H_0$, the feature is associated with the target.

### 2.3 Variance Threshold

**What it measures:** how much a feature varies across samples, with no reference to the target at all (fully **unsupervised**).

$$
\text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

A feature that barely moves (variance $\approx 0$) can't help separate anything — it's constant noise. Cheapest possible filter; usually the *first* pass on wide/high-dimensional data, before anything target-aware runs.

### 2.4 Pearson Correlation

**What it measures:** the strength and direction of a **linear** relationship between two numeric variables — used for two different jobs: feature-vs-target relevance, and feature-vs-feature redundancy.

$$
r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}
{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i - \bar{Y})^2}}
$$

$$
R^2 = r^2
$$

$r$ is always in $[-1, 1]$; $R^2$ is the fraction of variance explained.

### 2.5 ANOVA F-test

**What it measures:** whether a **numeric** feature's mean genuinely differs across the groups defined by a **categorical** target.

$$
F = \frac{\text{between-group variance}}{\text{within-group variance}}
= \frac{\sum_{i=1}^{k} n_i(\bar{x}_i - \bar{x})^2 / (k-1)}
{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 / (N-k)}
$$

With exactly 2 target groups, this is mathematically identical to a squared two-sample t-statistic ($F = t^2$) — ANOVA is the version that generalizes cleanly to 3+ group targets.

---

## 3. When to Use Which Method

The decision comes down to **feature type × target type**:

| Feature type | Target type | Use | Notes |
|---|---|---|---|
| Categorical | Categorical | **Chi-square** or **Information Gain** | Both work on the same contingency-table data; Info Gain is entropy-based, $\chi^2$ gives a p-value directly |
| Numeric | Categorical | **ANOVA F-test** | Compares group means; use Welch's variant if variances are unequal (see Levene's test) |
| Numeric | Numeric / ordinal | **Pearson Correlation** | Linear relevance only — use Spearman instead if the relationship is monotonic but non-linear |
| Any (any target, or no target at all) | — | **Variance Threshold** | Run this *first*, before any of the above, to cut obviously-dead features cheaply on wide data |
| Numeric ↔ Numeric (feature vs. feature, not vs. target) | — | **Pearson Correlation** | The redundancy/multicollinearity check — different job, same formula |

**Quick rule of thumb:** Variance Threshold first (cheap unsupervised pass) → Correlation for redundancy among survivors → Chi2/Info Gain/ANOVA depending on feature-target type combination for actual relevance ranking.

---

## 4. The Notebooks

| # | File | Dataset | Feature × Target | What it demonstrated |
|---|---|---|---|---|
| 01 | `01_Information_Gain.ipynb` | WineQT | categorical × categorical | Entropy-based relevance ranking |
| 02 | `02_Chi2_Test.ipynb` | Titanic | categorical × categorical | `Sex`/`Pclass` strongly associated with survival; `SibSp`/`Parch` flagged for sparse-cell caution |
| 03 | `03_Variance_Threshold.ipynb` | SECOM | numeric, unsupervised | 122 constant sensors removed outright; scale-correction needed for a meaningful non-zero threshold; blind-spot check caught 6 low-variance features that were still weakly target-relevant |
| 04 | `04_Pearson_Correlation.ipynb` | WineQT | numeric × numeric | Ranked relevance (`alcohol` strongest) + pruned redundant pairs (`fixed acidity` vs. `pH`/`density`/`citric acid`) with a traceable, target-aware rejection rule |
| 05 | `05_Anova_Test.ipynb` | Credit Default | numeric × categorical | Flagged `age` and `number_of_credit_accounts` as non-significant; **confirmed by training a model** — removing them left ROC-AUC/F1 essentially unchanged |

Each notebook follows the same skeleton: manual formula derivation → library cross-check (scipy/sklearn) → visualization → threshold decision with reasoning → caveats specific to that method.

---

## 5. Which Downstream Models Actually Care About This

Filter methods are model-agnostic in *how they score features*, but not every downstream model is equally hurt by skipping feature selection. Worth knowing which models make this step matter more:

**Models that care a lot:**
- **Linear/Logistic Regression** — redundant, correlated features inflate coefficient variance (multicollinearity) and destabilize the model; irrelevant features add noise directly to the decision boundary.
- **KNN, SVM (distance-based)** — every feature contributes to a distance calculation; irrelevant or unscaled features distort the neighborhood/margin directly.
- **Neural networks on small datasets** — more input dimensions without more signal just adds parameters to overfit with.

**Models that are more forgiving (but still benefit):**
- **Tree-based ensembles (Random Forest, XGBoost, LightGBM)** — naturally perform their own implicit feature selection via split-gain, so an irrelevant feature is often just ignored rather than actively harmful. Filtering still helps here mainly for training speed and interpretability, not accuracy.

This is exactly what the ANOVA notebook's before/after model check demonstrated directly: a logistic regression's performance survived removing two flagged features intact — the filter's judgment was validated by the model, not just asserted by a p-value.

---

## 6. Advantages & Disadvantages of Filter Methods (as a class)

**Advantages:**
- **Fast and cheap** — no model training required; scales to very wide data (e.g. SECOM's 590 features) where wrapper methods would be computationally infeasible.
- **Model-agnostic** — the same ranking can inform any downstream model choice.
- **Simple to interpret** — a p-value, an F-score, or a correlation coefficient is directly explainable to a non-technical stakeholder.
- **Good first-pass filter** — reduces dimensionality before a more expensive wrapper/embedded method runs on the survivors.

**Disadvantages:**
- **Univariate (mostly)** — evaluates each feature in isolation (except correlation's redundancy check), so it misses interactions between features that only matter jointly.
- **Ignores the downstream model entirely** — a feature that's weak by a univariate statistic could still be useful inside a specific model's interaction structure; filter methods can't see that.
- **Each test has its own blind spot** — $\chi^2$/ANOVA need adequate group sizes (sparse cells break them); correlation only sees linear relationships; variance threshold is blind to the target completely. None of these is a universal "importance" score — each answers a narrower question than it sounds like it does.
- **Threshold choice is still a judgment call** — every notebook in this series had to decide a p-value cutoff, an $|r|$ cutoff, or a variance cutoff by hand. Filter methods reduce the search, they don't remove the decision.

---

## 7. Final Summary

Filter methods are the fast, cheap, first layer of feature selection — five different statistical lenses (entropy, independence, variance, linear correlation, group-mean shift) each suited to a specific feature-type/target-type combination, run *before* any model is trained. Across this series:

- **Variance Threshold** cleared out the obviously dead weight on wide data (SECOM).
- **Chi-square** and **Information Gain** handled categorical-vs-categorical relevance (Titanic).
- **Pearson Correlation** did double duty — relevance *and* redundancy — on continuous features (WineQT).
- **ANOVA** handled numeric-vs-categorical relevance, and was the one method in this set actually **verified against a trained model** rather than left as a standalone statistic (Credit Default).

None of these methods is a replacement for the others — they answer different questions about "does this feature matter," and a real pipeline typically chains several of them (unsupervised filter → redundancy filter → relevance filter) before handing a reduced feature set to a wrapper or embedded method, or straight to a model. The consistent habit worth carrying forward from all five notebooks: **derive the statistic by hand once, cross-check against the library call, visualize before trusting a number, and — where possible, as ANOVA's notebook did — confirm the filter's judgment against an actual model rather than stopping at a p-value.**

"""Quick diagnostic: WHY is the PhiUSIIL baseline trivially perfect?

We measure, for each numeric feature on its own, how well it separates the two
classes (single-feature ROC-AUC). A feature near 1.00 is almost a giveaway.
Then we look at class-conditional stats for the top suspects to see the story.
"""
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import roc_auc_score

data = fetch_ucirepo(id=967)
X = data.data.features.copy()
y = (pd.to_numeric(data.data.targets.iloc[:, 0], errors="coerce") == 0).astype(int)  # 1 = phishing

X = X.drop(columns=[c for c in ["FILENAME", "URL", "Domain", "TLD", "Title"] if c in X.columns])
X = X.select_dtypes(include="number")

# Single-feature separability = how well this column alone ranks phishing vs legit.
rows = []
for col in X.columns:
    auc = roc_auc_score(y, X[col])
    rows.append((col, max(auc, 1 - auc)))  # direction-agnostic
sep = pd.DataFrame(rows, columns=["feature", "solo_auc"]).sort_values("solo_auc", ascending=False)

pd.set_option("display.width", 100)
print("Top 12 features by SINGLE-FEATURE AUC (1.00 = perfect separator on its own):")
print(sep.head(12).to_string(index=False))

# For the strongest content features, show class means and how often phishing = 0.
print("\nClass-conditional look at suspected 'empty phishing page' artifact:")
content = ["LineOfCode", "NoOfExternalRef", "NoOfImage", "NoOfJS", "NoOfCSS",
           "NoOfSelfRef", "HasSocialNet", "HasCopyrightInfo", "HasDescription"]
content = [c for c in content if c in X.columns]
summary = pd.DataFrame({
    "mean_legit":  X.loc[y == 0, content].mean(),
    "mean_phish":  X.loc[y == 1, content].mean(),
    "pct_phish_is_0": (X.loc[y == 1, content] == 0).mean() * 100,
    "pct_legit_is_0": (X.loc[y == 0, content] == 0).mean() * 100,
})
print(summary.round(2).to_string())

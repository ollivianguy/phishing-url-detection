r"""
failure_analysis.py
====================
Days 3-4 deliverable: WHERE does the honest (URL-only) model fail?

A high F1 you can't explain is worth less than a lower one you can. This script
finds the model's actual mistakes on the test set and shows you the real URLs
behind them, so you can write a couple of paragraphs on the failure modes.

It rebuilds the SAME url_only baseline as phishing_baseline.py (same SEED, same
split), then inspects:
  * false negatives  -> phishing URLs the model LET THROUGH  (the dangerous ones)
  * false positives  -> legitimate URLs it WRONGLY FLAGGED   (annoying for users)

Run it exactly like the baseline:
    .\.venv\Scripts\python.exe failure_analysis.py
"""

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SEED = 42
PHISHING_LABEL_IN_RAW = 0
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# --- Load, and keep the raw URL strings aside (for reading, NOT for training) ---
print("Loading data...")
data = fetch_ucirepo(id=967)
X_raw = data.data.features.copy()
urls = X_raw["URL"]                                    # keep the actual URLs
y = (pd.to_numeric(data.data.targets.iloc[:, 0], errors="coerce")
     == PHISHING_LABEL_IN_RAW).astype(int)             # 1 = phishing
X = X_raw[URL_ONLY_FEATURES].copy()

# --- Same split as the baseline (same SEED => identical test rows) ---
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

print("Training random forest...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
rf.fit(X_train, y_train)

# --- Predictions on the test set, with the model's confidence ---
phish_col = list(rf.classes_).index(1)                 # column for P(phishing)
proba = rf.predict_proba(X_test)[:, phish_col]
pred = (proba >= 0.5).astype(int)
truth = y_test.to_numpy()

# One tidy table: features + truth + prediction + confidence + the real URL.
res = X_test.copy()
res["true"] = truth
res["pred"] = pred
res["proba_phish"] = proba
res["url"] = urls.loc[X_test.index].to_numpy()

false_neg = res[(res["true"] == 1) & (res["pred"] == 0)]   # phishing slipped through
false_pos = res[(res["true"] == 0) & (res["pred"] == 1)]   # legit wrongly flagged
caught    = res[(res["true"] == 1) & (res["pred"] == 1)]   # phishing correctly caught

n_phish = int((truth == 1).sum())
n_legit = int((truth == 0).sum())
print("\n" + "=" * 66)
print(f"Test set: {len(res):,} URLs  ({n_phish:,} phishing, {n_legit:,} legit)")
print(f"  False negatives (phishing MISSED):  {len(false_neg):>4}  "
      f"= {len(false_neg)/n_phish:.2%} of phishing")
print(f"  False positives (legit FLAGGED):    {len(false_pos):>4}  "
      f"= {len(false_pos)/n_legit:.2%} of legit")
print("=" * 66)

# --- What makes a phishing URL slip through? Compare MISSED vs CAUGHT phishing ---
compare_cols = ["IsHTTPS", "URLLength", "NoOfSubDomain",
                "NoOfOtherSpecialCharsInURL", "DegitRatioInURL"]
print("\nMissed vs caught phishing (mean feature values):")
print(f"  {'feature':<28}{'missed':>10}{'caught':>10}")
for c in compare_cols:
    print(f"  {c:<28}{false_neg[c].mean():>10.3f}{caught[c].mean():>10.3f}")
print("  (If 'missed' phishing look more like legit URLs -- e.g. higher IsHTTPS,")
print("   shorter, fewer special chars -- that is your failure-mode story.)")

print(f"\nModel confidence on its mistakes (proba_phish, 0.5 = the cutoff):")
print(f"  missed phishing : mean {false_neg['proba_phish'].mean():.3f}  "
      f"(were these borderline, or confidently wrong?)")
print(f"  false alarms    : mean {false_pos['proba_phish'].mean():.3f}")

# --- Show real examples and save them for the writeup ---
show = ["url", "proba_phish", "IsHTTPS", "URLLength", "NoOfOtherSpecialCharsInURL"]
pd.set_option("display.max_colwidth", 80)
print("\n--- Sample phishing URLs the model MISSED (false negatives) ---")
print(false_neg[show].head(15).to_string(index=False))
print("\n--- Sample legit URLs the model WRONGLY FLAGGED (false positives) ---")
print(false_pos[show].head(15).to_string(index=False))

false_neg[show].to_csv("failures_false_negatives.csv", index=False)
false_pos[show].to_csv("failures_false_positives.csv", index=False)
print("\nSaved full lists -> failures_false_negatives.csv, failures_false_positives.csv")
print("\nNow WRITE 1-2 paragraphs: what do the misses have in common? What kind of")
print("legit URLs trip the alarm? That is your 'I can explain my model' deliverable.")

r"""
diagnose.py
===========
Data audit: WHY does a model trained on all of PhiUSIIL's features score ~100%?

Prints the evidence (and saves it to results/):
  1. several columns separate the classes almost perfectly ON THEIR OWN;
  2. URLSimilarityIndex is an answer key -- EVERY legitimate URL scores exactly
     100 on it, so a one-line rule catches nearly all phishing;
  3. page-content features are ~0 for phishing: those pages were captured
     nearly empty, so "has content" quietly stands in for "is legitimate";
  4. the two classes weren't even processed identically: the URL's final
     character was dropped before feature extraction for every legitimate URL
     but only about half of the phishing ones;
  5. every legitimate URL is written in one normalised form, 'https://www.' --
     a subtler shortcut that attack.py and defend.py show an attacker can copy
     for free;
  6. otherwise the 21 URL-only features look clean -- none is a giveaway on its
     own, and TLDLegitimateProb is an outside score, not a disguised label.

Run:
    .\.venv\Scripts\python.exe diagnose.py
"""
import numpy as np
import pandas as pd

from common import ID_LIKE_COLUMNS, URL_ONLY_FEATURES, dropped_chars, load_data, save_table, solo_auc

CONTENT_FEATURES = ["LineOfCode", "NoOfExternalRef", "NoOfImage", "NoOfJS", "NoOfCSS",
                    "NoOfSelfRef", "HasSocialNet", "HasCopyrightInfo", "HasDescription"]


def main():
    X, y = load_data()
    phish, legit = (y == 1).to_numpy(), (y == 0).to_numpy()
    print(f"Dataset: {len(X):,} URLs -- {legit.sum():,} legitimate, "
          f"{phish.sum():,} phishing ({phish.mean():.1%} phishing)")

    numeric = X.drop(columns=[c for c in ID_LIKE_COLUMNS if c in X.columns]).select_dtypes("number")

    # 1. Single-feature separability -------------------------------------------
    auc = solo_auc(numeric, y)
    print("\n1. Features that separate the classes ON THEIR OWN (AUC; 1.00 = giveaway):")
    print(auc.head(8).round(3).to_string())
    save_table(auc.rename("solo_auc").round(4).to_frame(), "data_audit_solo_auc.csv",
               index_label="feature")

    # 2. URLSimilarityIndex is an answer key -----------------------------------
    usi = X["URLSimilarityIndex"].to_numpy()
    usi_corr = abs(float(np.corrcoef(usi, y)[0, 1]))
    print(f"\n   (Its correlation with the label is only {usi_corr:.2f} -- a correlation check "
          "would wave it through. Single-feature AUC catches it.)")
    flagged = usi < 100                       # the rule: "not 100 => phishing"
    rule_accuracy = float((flagged == phish).mean())
    rule_recall = float(flagged[phish].mean())
    rule_false_alarms = int(flagged[legit].sum())
    print("\n2. URLSimilarityIndex is an answer key:")
    print(f"   legitimate URLs scoring exactly 100: {np.mean(usi[legit] == 100):.2%}"
          f"  (lowest legitimate score: {usi[legit].min():.1f})")
    print(f"   phishing URLs scoring exactly 100:   {np.mean(usi[phish] == 100):.2%}")
    print(f"   The one-line rule 'if it isn't 100, it's phishing' is right {rule_accuracy:.2%}"
          f" of the time:\n   it catches {rule_recall:.2%} of phishing with "
          f"{rule_false_alarms} false alarms.")

    # 3. Phishing pages were captured nearly empty ------------------------------
    content = pd.DataFrame({
        "mean_legit": numeric.loc[legit, CONTENT_FEATURES].mean(),
        "mean_phishing": numeric.loc[phish, CONTENT_FEATURES].mean(),
        "pct_phishing_zero": (numeric.loc[phish, CONTENT_FEATURES] == 0).mean() * 100,
        "pct_legit_zero": (numeric.loc[legit, CONTENT_FEATURES] == 0).mean() * 100,
    })
    print("\n3. Phishing pages were captured nearly empty:")
    print(content.round(2).to_string())
    save_table(content.round(4), "data_audit_page_content.csv", index_label="feature")

    # 4. The classes were processed differently -------------------------------
    dropped = (dropped_chars(X) > 0).to_numpy()
    print("\n4. The URL's final character was dropped before feature extraction for:")
    print(f"   {dropped[legit].mean():.2%} of legitimate URLs, but only "
          f"{dropped[phish].mean():.2%} of phishing URLs.")
    print("   (A string-handling slip in the original pipeline, applied unevenly across classes.)")

    # 5. Every legitimate URL is written 'https://www.' ------------------------
    url = X["URL"].astype(str).str.lower()
    https, www = url.str.startswith("https://").to_numpy(), url.str.match(r"^https?://www\.").to_numpy()
    print("\n5. Every legitimate URL is written in one normalised form:")
    print(f"   starts 'https://':  legitimate {https[legit].mean():.2%}   phishing {https[phish].mean():.2%}")
    print(f"   has a 'www.' host:  legitimate {www[legit].mean():.2%}   phishing {www[phish].mean():.2%}")
    print("   Both are free for an attacker to copy -- see attack.py and defend.py.")

    # 6. The honest URL-only feature set ---------------------------------------
    url_auc = auc[URL_ONLY_FEATURES].sort_values(ascending=False)
    per_tld = (pd.DataFrame({"tld": X["TLD"], "prob": X["TLDLegitimateProb"], "legit": legit})
               .groupby("tld").agg(prob=("prob", "first"), legit_rate=("legit", "mean"),
                                   n=("legit", "size")))
    common_tlds = per_tld[per_tld["n"] >= 50]
    tld_r = float(np.corrcoef(common_tlds["prob"], common_tlds["legit_rate"])[0, 1])
    print("\n6. Otherwise, the 21 URL-only features look clean:")
    print(f"   strongest on its own: {url_auc.index[0]} (AUC {url_auc.iloc[0]:.3f}) -- no "
          "single giveaway; a model has to combine many weak signals.")
    print(f"   TLDLegitimateProb vs this dataset's own per-TLD legit rate: r = {tld_r:.2f}, "
          "so it is an outside score, not a disguised label.")

    summary = pd.Series({
        "urls": len(X), "legitimate": int(legit.sum()), "phishing": int(phish.sum()),
        "usi_label_correlation": round(usi_corr, 4),
        "usi_legit_share_100": float(np.mean(usi[legit] == 100)),
        "usi_phishing_share_100": float(np.mean(usi[phish] == 100)),
        "one_line_rule_accuracy": rule_accuracy,
        "one_line_rule_recall": rule_recall,
        "one_line_rule_false_alarms": rule_false_alarms,
        "last_char_dropped_legit": float(dropped[legit].mean()),
        "last_char_dropped_phishing": float(dropped[phish].mean()),
        "https_share_legit": float(https[legit].mean()),
        "https_share_phishing": float(https[phish].mean()),
        "www_share_legit": float(www[legit].mean()),
        "www_share_phishing": float(www[phish].mean()),
        "url_only_strongest_feature": url_auc.index[0],
        "url_only_strongest_solo_auc": round(float(url_auc.iloc[0]), 4),
        "tldprob_vs_legit_rate_r": round(tld_r, 4),
    })
    save_table(summary.to_frame("value"), "data_audit_summary.csv", index_label="metric")
    print("\nSaved -> results/data_audit_*.csv")


if __name__ == "__main__":
    main()

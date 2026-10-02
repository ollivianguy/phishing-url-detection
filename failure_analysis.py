r"""
failure_analysis.py
====================
Days 3-4: WHERE does the honest (URL-only) model fail -- and why?

A high F1 you can't explain is worth less than a lower one you can. This
rebuilds the url_only random forest (same seed, same split as every script),
then looks at the real URLs behind its mistakes on the test set:
  * false negatives -- phishing it LET THROUGH (the dangerous ones);
  * false positives -- legitimate sites it WRONGLY FLAGGED (the annoying ones).

It uses the model's default decision rule (flag when P(phishing) > 0.5): this is
the model as trained, before the attack stage tunes every model to a fixed
false-alarm rate.

Run:
    .\.venv\Scripts\python.exe failure_analysis.py
"""
import pandas as pd

from common import URL_ONLY_FEATURES, load_data, phish_proba, save_table, split, train_forest

COMPARE = ["IsHTTPS", "HasWWW", "URLLength", "NoOfOtherSpecialCharsInURL", "DegitRatioInURL"]
SHOW = ["URL", "proba_phish", "IsHTTPS", "URLLength", "NoOfOtherSpecialCharsInURL"]


def main():
    X, y = load_data()
    X_train, _, X_test, y_train, _, y_test = split(X, y)
    forest = train_forest(X_train[URL_ONLY_FEATURES], y_train)

    res = X_test[["URL"] + URL_ONLY_FEATURES].copy()
    res["HasWWW"] = res["URL"].str.lower().str.match(r"^https?://www\.").astype(int)   # for reading only
    res["truth"] = y_test.to_numpy()
    res["pred"] = forest.predict(X_test[URL_ONLY_FEATURES])
    res["proba_phish"] = phish_proba(forest, X_test[URL_ONLY_FEATURES]).round(3)

    missed = res[(res["truth"] == 1) & (res["pred"] == 0)]        # phishing let through
    false_alarm = res[(res["truth"] == 0) & (res["pred"] == 1)]   # legit wrongly flagged
    caught = res[(res["truth"] == 1) & (res["pred"] == 1)]
    n_phish, n_legit = int((res["truth"] == 1).sum()), int((res["truth"] == 0).sum())

    print("=" * 68)
    print(f"Test set: {len(res):,} URLs ({n_phish:,} phishing, {n_legit:,} legitimate)")
    print(f"  phishing MISSED (false negatives):  {len(missed):>4} = {len(missed) / n_phish:.2%} of phishing")
    print(f"  legit FLAGGED (false positives):    {len(false_alarm):>4} = {len(false_alarm) / n_legit:.2%} of legit")
    print("=" * 68)

    comparison = pd.DataFrame({"missed_phishing": missed[COMPARE].mean(),
                               "caught_phishing": caught[COMPARE].mean(),
                               "false_alarms": false_alarm[COMPARE].mean()})
    print("\nWhat do the mistakes have in common? (mean feature values)")
    print(comparison.round(3).to_string())
    print(f"\nModel confidence on its mistakes (P(phishing); the line is at 0.5):")
    print(f"  missed phishing: mean {missed['proba_phish'].mean():.3f}  -> confidently wrong")
    print(f"  false alarms:    mean {false_alarm['proba_phish'].mean():.3f}  -> borderline calls")

    pd.set_option("display.max_colwidth", 70)
    print("\n--- Phishing the model MISSED ---")
    print(missed[SHOW].head(12).to_string(index=False))
    print("\n--- Legitimate sites the model FLAGGED ---")
    print(false_alarm[SHOW].head(12).to_string(index=False))

    save_table(missed[SHOW], "failures_false_negatives.csv", index=False)
    save_table(false_alarm[SHOW], "failures_false_positives.csv", index=False)
    summary = comparison.round(4)
    summary.loc["count"] = [len(missed), len(caught), len(false_alarm)]
    summary.loc["mean_proba_phish"] = [missed["proba_phish"].mean(), caught["proba_phish"].mean(),
                                       false_alarm["proba_phish"].mean()]
    save_table(summary.round(4), "failure_summary.csv", index_label="statistic")
    print("\nSaved -> results/failures_*.csv, results/failure_summary.csv")
    print("The pattern to write up: the model learned what phishing URLs usually LOOK like")
    print("(long, messy, plain HTTP). The phishing it misses looks like 'https://www.<name>' --")
    print("exactly how every legitimate URL in this dataset is written.")


if __name__ == "__main__":
    main()

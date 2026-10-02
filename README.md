# Phishing URL Detection: Leakage, Realistic Attacks & Honest Evaluation

A machine-learning phishing-URL detector that scores 99.7% F1 — and an
investigation into why that number is far too good to trust, how easily a
realistic attacker gets around it, and what defending it actually costs.

> **Status:** complete and reproducible. One command regenerates every number,
> table and figure below (`run_all.py`, a few minutes), and 15 tests check the
> URL feature engine the attack depends on.

---

## TL;DR

- **The dataset gives the answer away.** On PhiUSIIL (235,795 URLs), a model
  using every feature scores a perfect 1.000. Every legitimate URL scores
  exactly 100 on `URLSimilarityIndex`, so a one-line rule catches 99.2% of
  phishing with zero false alarms — and phishing pages were crawled nearly empty.
- **The "honest" URL-only model still leans on a shortcut.** Using only features
  computed from the URL string gives F1 0.997 — but every legitimate URL in the
  dataset is written `https://www.…`, and the model's #1 signal is `IsHTTPS`.
- **A realistic attacker gets half the phishing through.** I reverse-engineered
  the dataset's feature rules so the attack edits *real URLs* (rules checked
  against all 235,795 rows). Serving the page over HTTPS alone drops recall from
  99.4% to 67.7%; an attacker who tries every combination of five realistic moves
  — only on domains they actually own — gets **48.9%** of phishing past it.
- **Feature-space attacks mislead.** Writing the same moves straight into the
  feature vector "evades" 69% — but **100%** of those vectors are impossible for
  any real URL.
- **The obvious defences don't work, and the honest one is humbling.** Held to
  the same false-alarm rate (1 in 1,000), adversarial training and deleting the
  `IsHTTPS` column barely help: the scheme can be rebuilt from the other features
  for 99.7% of URLs. Removing that information properly makes the free moves
  useless — but recall on *unattacked* phishing falls from 99.4% to 67.7%. Most of
  the model's apparent skill was the `https://www.` artifact.

---

## Why this project is about honesty, not leaderboard scores

A phishing detector reporting 99% sounds great until you ask what the number
rests on. Four ideas matter more here than a high score:

1. **The base-rate problem** — accuracy is the wrong metric for imbalanced data.
2. **Data leakage** — a "perfect" model can be reading an answer key.
3. **Explainable failure** — being able to say *why* the model makes its mistakes.
4. **Realistic robustness** — whether the model survives an attacker who edits
   the URL, measured in a way that could actually happen.

**Data.** The PhiUSIIL Phishing URL dataset (UCI ML Repository, id 967): 134,850
legitimate and 100,945 phishing URLs (42.8% phishing). I flip the label so
phishing = 1, the positive class that precision and recall describe. Split
70 / 15 / 15 (stratified): choices are made on validation, and the test set is
only ever used to report a final score.

**Why not accuracy?** A lazy model that always says "legitimate" scores 57.2%
accuracy here while catching zero phishing. Accuracy rewards guessing the common
class, so it can hide a useless model — and on real traffic, where phishing is
well under 1% of URLs, it becomes almost meaningless. I report precision, recall,
F1 and ROC-AUC, with phishing as the positive class.

---

## 1. The dataset gives the answer away

Ranking every feature by how well it separates the classes **on its own**
(single-feature ROC-AUC, where 1.00 is a perfect giveaway):

| feature | solo AUC |
|---|---|
| URLSimilarityIndex | 0.996 |
| LineOfCode | 0.990 |
| NoOfExternalRef | 0.988 |
| NoOfImage | 0.980 |
| NoOfJS | 0.971 |

Correlation would have missed the top one (its correlation with the label is only
0.86), so single-feature AUC is the right tool here. Digging in found four
data-collection artifacts:

- **`URLSimilarityIndex` is an answer key.** All 134,850 legitimate URLs score
  exactly 100 on it; only 0.78% of phishing do. The rule *"if it isn't 100, it's
  phishing"* is right 99.67% of the time and never raises a false alarm.
- **Phishing pages were captured nearly empty**, so "has page content" stands in
  for "is legitimate":

  | feature | mean (legit) | mean (phishing) | phishing rows at 0 |
  |---|---|---|---|
  | LineOfCode | 1,947 | 66 | — |
  | NoOfImage | 45 | 0.9 | 83% |
  | NoOfJS | 18 | 0.9 | 77% |
  | HasSocialNet | 0.79 | 0.01 | 99% |

- **The classes weren't processed identically.** The URL's final character was
  dropped before feature extraction for 100% of legitimate URLs but only 52% of
  phishing ones — a string-handling slip in the original pipeline.
- **Every legitimate URL is written `https://www.…`** (100%), against 48.7% HTTPS
  and 41.4% `www.` among phishing. This one is subtle: both are free for an
  attacker to copy, which sections 4 and 5 turn out to be all about.

![All-features model: the top signals are the leaky ones](figures/feature_importance_all_features.png)

**Decision:** the honest baseline uses only the 21 features computable from the
URL string — the part an attacker controls. None of them is a giveaway on its
own (the strongest has a solo AUC of 0.82), and `TLDLegitimateProb` is an outside
score rather than a disguised label (correlation 0.07 with each TLD's legit rate).

---

## 2. Baselines

The chosen model on the held-out test set (logistic regression vs random forest,
picked on validation):

| baseline | precision | recall | F1 | ROC-AUC |
|---|---|---|---|---|
| all 50 features *(leaky)* | 1.000 | 1.000 | 1.000 | 1.000 |
| URL string only, 21 features | 0.998 | 0.995 | 0.997 | 0.999 |

![What the URL-only model relies on](figures/feature_importance_url_only.png)

`IsHTTPS` is the model's #1 signal by both methods: 37% of the built-in
importance, and shuffling it costs more F1 than any other feature. The built-in
measure is biased towards some feature types, so permutation importance is the
check that the attack in section 4 is aimed at the right thing.

---

## 3. Where the URL-only model fails

On the test set it misses 70 phishing URLs (0.46%) and flags 30 legitimate ones
(0.15%). The mistakes are not random:

| | missed phishing | caught phishing | false alarms |
|---|---|---|---|
| starts `https://` | **100%** | 49% | 100% |
| has `www.` | **100%** | 41% | 100% |
| mean URL length | 28 | 46 | 30 |
| mean symbols (`-`, `.`, …) | 1.5 | 3.9 | 1.7 |

Every phishing URL that slips through looks like `https://www.<name>` — e.g.
`https://www.jp-metamask.org`, `https://www.notepad-install.top` — exactly how
every legitimate URL in the dataset is written, and the model is *confident* they
are safe (mean phishing probability 0.13). The false alarms are borderline calls
(0.62) on legitimate sites that look "messy": `commonfuture-paris2015.org`,
`study-in-egypt.gov.eg`. **The model learned what phishing URLs usually look
like, not what makes them malicious** — which tells you exactly how to attack it.

---

## 4. Attacking it: feature space vs problem space

**How the attack works**

- **Real URLs, not made-up numbers.** The dataset's feature rules aren't
  published, so I reverse-engineered them (e.g. character counts skip
  `http(s)://` and `www.`; `CharContinuationRate` is the longest runs of letters,
  digits and symbols divided by the domain-name length). 19 of the 21 features
  reproduce the stored values for at least 98.6% of all 235,795 rows (6 of them on
  every row); `URLLength` follows from each row's own length, and `URLCharProb`
  is approximated (r = 0.985). Each attack edits the actual URL and recomputes
  its features with these rules.
- **Five realistic moves:** serve over HTTPS (free certificates) · add `www.` (a
  subdomain of your own domain) · a name without digits or hyphens · register as
  `.com` · put the page at the site root.
- **Only on domains the attacker owns.** Half the phishing URLs (50.2%) sit on
  someone else's domain — a hosting platform (`web.app`, `firebaseapp.com`), an
  IPFS gateway or link shortener (`ipfs.io`, `bit.ly`), a compromised site, an IP
  address. For those, only the HTTPS switch is allowed.
- **A realistic operating point.** The threshold is set on validation data so
  that at most 0.1% of legitimate URLs are flagged — 1 in 1,000, the way a real
  product is tuned.
- **An adaptive attacker.** For each URL it tries all 32 combinations of moves and
  keeps whichever the model finds least suspicious, as someone testing variants
  against a detector before launching would.

**Results** — on all 15,142 test phishing URLs (95% intervals within ±0.8 points):

| attack | phishing caught | evasion |
|---|---|---|
| no attack | 99.4% | 0.6% |
| HTTPS only | 67.7% | 32.3% |
| all five moves at once | 54.5% | 45.5% |
| **adaptive (best of 32 per URL)** | **51.1%** | **48.9%** |
| feature-space (same five moves) | 30.9% | 69.1% |

![How much phishing the URL-only model still catches under attack](figures/attack_recall.png)

- **One free move does most of the damage.** HTTPS alone gets a third of phishing
  past the detector, and it appears in 79% of the adaptive attacker's successes
  (`www.` in 27%). 7,313 URLs went from caught to evading — e.g.
  `http://www.pradopro.ru` scored 1.00 as phishing; `https://www.pradopro.ru`
  scores 0.00.
- **Feature-space vs problem-space.** Writing the same five moves straight into
  the feature vector claims 69% evasion — 20 points more than the realistic
  attacker manages. And **100%** of those vectors break a rule that every real
  URL obeys (e.g. "zero symbols" while the domain still contains dots), whereas
  0% of the problem-space vectors do — the same as genuine URLs. The feature-space
  number describes URLs that cannot exist. That gap is the point of Pierazzi et
  al. (2020): attacks have to survive the constraints of the real object.

---

## 5. Defending it — and what each defence costs

Every model is held to the **same** false-alarm rate (0.1% on legitimate
validation URLs), and the attacker **adapts to each defended model**, trying all
32 move combinations against it rather than replaying the attack that beat the
original. Adversarial training only ever sees two of the five moves (HTTPS and a
cleaner name), so the other three test whether it learned a general lesson.

| model | no attack | HTTPS only | adaptive, anticipated moves | **adaptive, all moves** | cost on normal traffic |
|---|---|---|---|---|---|
| baseline | 99.4% | 67.7% | 66.3% | **51.1%** | — |
| adversarial training | 95.7% | 68.0% | 67.5% | **51.3%** | −3.8 pts |
| delete the `IsHTTPS` column | 99.2% | 70.4% | 67.9% | **53.3%** | −0.3 pts |
| prefix-blind features | 67.7% | 67.7% | 65.1% | **54.8%** | −31.8 pts |

![Each defence against an attacker who adapts to it](figures/defence_robustness.png)

**Deleting the column doesn't delete the information.** The dataset counts
characters *after* `http(s)://` but measures `URLLength` *with* it, so
`URLLength` minus the counted characters is exactly 7, 8, 11 or 12 — the length of
`http://`, `https://`, `http://www.` or `https://www.` — for 99.8% of URLs. From
that gap alone, HTTPS can be recovered for 99.7% of URLs, so the model still
"sees" the scheme.

**Adversarial training buys nothing at a fair false-alarm rate.** It made the
model suspicious of exactly the kind of URL legitimate sites use, so its
threshold had to rise from 0.62 to 0.98 to keep false alarms at 1 in 1,000 — and
it ends up catching 3.8 points less unattacked phishing for a 0.2-point gain
under attack. At the naive default threshold of 0.5 it *looks* like it works
(74.4% caught under attack), but only because it flags **9.7% of legitimate
sites**, 65 times as many as the baseline. Comparing at a fixed false-alarm rate
is what stops a model buying "robustness" by flagging everything.

**Removing the information works — and reveals the real problem.** "Prefix-blind"
features drop `IsHTTPS` *and* measure every length and ratio without
`http(s)://www.`, so the two free moves no longer change anything (the attacker
used them in 0% of its successes). But recall on normal, unattacked phishing falls
from 99.4% to 67.7%, and ROC-AUC from 0.999 to 0.933. That is the honest measure
of what these URL features can do once you stop rewarding a shortcut any attacker
copies for free: most of the original model's skill was the `https://www.`
artifact.

**The trade-off, in one sentence:** making the model unable to be fooled by free
moves cost about a third of its detection rate on everyday traffic — and even
then, an attacker willing to register a cleaner name or a `.com` still gets 45% of
phishing through.

---

## How the conclusions changed when the attack got honest

My first version of this project concluded the opposite: that both defences
restored about 99% robustness for almost no cost. Two mistakes produced that:

1. **The attack's feature extractor didn't match the dataset's.** It counted
   characters over the whole URL, while the dataset skips `http(s)://` and
   `www.`. So switching to HTTPS changed `URLLength` and the letter count
   *together*, which hid the leak through `URLLength`. It also generated some
   URLs that cannot exist (an IP address with its digits stripped became
   `https://.../`), and "re-registered" phishing hosted on `web.app` or `ipfs.io`
   under domains the attacker could never own.
2. **Defences were compared at the default 0.5 threshold**, where adversarial
   training looks robust simply because it flags far more legitimate sites.

Checking the feature engine against all 235,795 rows, restricting moves to what
an attacker can really do, and fixing the false-alarm rate flipped the
conclusion. That is the clearest lesson of the project: **an evaluation is only
as honest as its attack.**

---

## Reproduce it

Requires **Python 3.13** (the pinned versions ship prebuilt wheels for it). From
the project folder:

```bash
# one-time setup -- Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# one-time setup -- macOS / Linux
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# regenerate every number, table and figure (a few minutes)
.\.venv\Scripts\python.exe run_all.py          # or ./.venv/bin/python run_all.py

# run the tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The dataset downloads from UCI on the first run and is cached in `data/`. Each
stage can also be run on its own:

| script | what it does |
|---|---|
| `diagnose.py` | Data audit: the leakage evidence in section 1 |
| `phishing_baseline.py` | Both baselines, metrics, confusion matrices, feature importances |
| `failure_analysis.py` | The URL-only model's mistakes, with the real URLs |
| `attack.py` | Problem-space vs feature-space attacks, at a fixed false-alarm rate |
| `defend.py` | The three defences against an adaptive attacker, and their cost |
| `common.py` | Shared code: data loading, the split, and the URL feature engine |

Every figure in `figures/` has its numbers in `results/` as CSV.

---

## Limitations and what I'd do next

- **The legitimate class isn't realistic.** Every legitimate URL is a homepage
  written `https://www.<domain>`, so any model trained on this data learns
  "homepage-shaped means safe". The most valuable next step is rebuilding the
  legitimate class the way real traffic looks (deep links, sites without `www.`)
  and re-running everything.
- **The attack is approximate in places.** `URLCharProb` is approximated
  (r = 0.985); ownership is inferred from how many URLs share a domain, which
  treats a few heavily reused attacker domains as shared (making the attack
  slightly conservative); and it assumes a cleaner name or a `.com` is available
  to register.
- **URL-only features are a narrow view.** Real detectors also use signals that
  are expensive to fake — domain age, certificate history, hosting reputation.
  Adding those, then re-running the adaptive attack, is the natural next defence.
- **Scope.** One dataset, one model family for the attack and defences, one
  seed, and one round of adversarial training (iterating it against the adaptive
  attacker is worth trying, although the overlap with legitimate URLs suggests
  the limit is fundamental).

---

## Data and references

- **Dataset:** PhiUSIIL Phishing URL dataset, UCI Machine Learning Repository
  (id 967), https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset.
  Introduced in A. Prasad and S. Chandra, "PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning", *Computers & Security* (2024),
  https://doi.org/10.1016/j.cose.2023.103545.

*Background reading — read these before citing them:*

- Pierazzi et al., *Intriguing Properties of Adversarial ML Attacks in the Problem
  Space* (2020) — the feature-space vs problem-space distinction.
- Carlini et al., *On Evaluating Adversarial Robustness* (2019) and Tramèr et al.,
  *On Adaptive Attacks to Adversarial Example Defenses* (2020) — why a defence
  must be tested against an attacker who adapts to it.
- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014), and
  Szegedy et al., *Intriguing Properties of Neural Networks* (2013) — where
  adversarial examples started.

## Repo layout

```
common.py              shared code: data, split, models, URL feature engine
diagnose.py            1. data audit (leakage evidence)
phishing_baseline.py   2. baselines
failure_analysis.py    3. failure modes
attack.py              4. attacks
defend.py              5. defences
run_all.py             all five stages, in order
tests/                 checks for the feature engine and helpers
figures/  results/     every plot, and the numbers behind it
requirements.txt       pinned dependencies (Python 3.13)
```

"""
common.py
=========
Shared building blocks for every script in this project, so each idea lives in
exactly one place:

  * constants -- the seed, the feature lists, the label encoding, and the
    "operating point" every model is compared at;
  * loading PhiUSIIL (downloaded once, then read from a local copy);
  * the train / validation / test split and the random-forest recipe;
  * the URL FEATURE ENGINE -- edit a real URL, then recompute its 21 features
    with the dataset's own rules. This is the heart of the problem-space attack;
  * a small, consistent chart style.

Nothing here runs on its own; the scripts import from it.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"            # local copy of the dataset (git-ignored)
FIG_DIR = ROOT / "figures"          # every plot the project makes
RESULTS_DIR = ROOT / "results"      # every table of numbers (each figure's data too)
CACHE_FILE = DATA_DIR / "phiusiil.csv.gz"

UCI_ID = 967
SEED = 42

# PhiUSIIL stores the label as 0 = phishing, 1 = legitimate. We flip it so that
# 1 = phishing everywhere: phishing is what we want to catch, so it should be the
# "positive" class that precision and recall describe.
PHISHING_LABEL_IN_RAW = 0

# Identifiers and free text. Memorising one specific URL or page title is not a
# signal that generalises, so these never go into a model.
ID_LIKE_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# The 21 features computable from the URL string alone: no page download, and
# exactly the part an attacker controls. URLSimilarityIndex is deliberately left
# out -- every legitimate URL in the dataset scores exactly 100 on it, so it is
# an answer key, not a signal (see diagnose.py).
URL_ONLY_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# The OPERATING POINT used whenever models are compared under attack. Each
# model's decision threshold is chosen on the VALIDATION set so that at most
# 0.1% of legitimate URLs (1 in 1,000) get flagged -- the way a real product is
# tuned. Without a shared false-alarm rate, a model could look "more robust"
# simply by flagging more of everything.
TARGET_FPR = 0.001


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _raw_frame():
    """The dataset as UCI serves it (features + raw label), downloaded once."""
    if not CACHE_FILE.exists():
        from ucimlrepo import fetch_ucirepo   # only needed for the first download
        print(f"Downloading PhiUSIIL (UCI id {UCI_ID}) -- one time only; "
              f"a copy is kept in {CACHE_FILE.relative_to(ROOT)}.")
        data = fetch_ucirepo(id=UCI_ID)
        frame = data.data.features.copy()
        frame["label"] = data.data.targets.iloc[:, 0].to_numpy()
        DATA_DIR.mkdir(exist_ok=True)
        frame.to_csv(CACHE_FILE, index=False)
    # Always read the saved copy, so the first run and every later run see
    # byte-for-byte the same data.
    return pd.read_csv(CACHE_FILE)


def load_data():
    """Return (X, y): every column except the label, and y with 1 = phishing."""
    frame = _raw_frame()
    y = (pd.to_numeric(frame["label"], errors="coerce") == PHISHING_LABEL_IN_RAW).astype(int)
    y.name = "is_phishing"
    return frame.drop(columns="label").copy(), y.copy()


def split(X, y):
    """Stratified 70 / 15 / 15 train / validation / test split.

    'stratify=y' keeps the same phishing rate in every part, and the fixed seed
    gives every script exactly the same rows. The validation set is where we
    are allowed to make choices (which model, which threshold); the test set is
    only ever used to report a final score, so that score stays honest."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Models, thresholds, and small statistics helpers
# ---------------------------------------------------------------------------
def train_forest(X, y):
    """The project's random-forest recipe (identical settings everywhere)."""
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X, y)


def phish_proba(model, X):
    """P(phishing) for each row. Looks the column up instead of assuming it.

    Rounded to 9 decimals: a forest adds up its trees' votes in parallel, in
    whatever order the threads finish, which can leave ~1e-16 of noise -- enough
    to break ties differently for two IDENTICAL inputs. Its probabilities are
    multiples of 1/300 anyway, so rounding loses nothing."""
    return np.round(model.predict_proba(X)[:, list(model.classes_).index(1)], 9)


def threshold_at_fpr(scores_on_legit, target_fpr=TARGET_FPR):
    """The lowest threshold that flags at most `target_fpr` of these legit URLs.

    A URL is flagged when its P(phishing) is strictly greater than the
    threshold. Lowest = catches as much phishing as the false-alarm budget allows."""
    s = np.sort(np.asarray(scores_on_legit, dtype=float))[::-1]   # most suspicious first
    allowed = int(np.floor(target_fpr * len(s)))                  # false alarms we can afford
    return float(s[allowed]) if allowed < len(s) else -np.inf


def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a proportion (Wilson score method)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def solo_auc(X, y):
    """How well each column ALONE separates the classes (ROC-AUC, either way round).

    1.00 = a perfect giveaway on its own; 0.50 = useless on its own. Unlike
    correlation, this also catches a feature whose relationship with the label
    is strong but not a straight line."""
    out = {}
    for col in X.columns:
        auc = roc_auc_score(y, X[col])
        out[col] = max(auc, 1 - auc)
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------
def save_figure(fig, name):
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_table(table, name, index=True, index_label=None):
    """Save a table to results/. Every figure has one of these as its plain-text twin."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    table.to_csv(path, index=index, index_label=index_label)
    return path


# ---------------------------------------------------------------------------
# The URL feature engine
# ---------------------------------------------------------------------------
# To attack the model with REAL URLs we must turn an edited URL back into the
# 21 numbers the model sees -- using the same rules the dataset's authors used.
# Those rules are not published, so they were reverse-engineered from the data
# and checked against all 235,795 rows (URLFeatureEngine.check_rules):
#
#   * character counts skip the leading "http(s)://" and "www.";
#   * the three ratio features divide those counts by URLLength;
#   * the dataset sometimes dropped a URL's final character before counting
#     (for every legitimate URL, and about half the phishing ones), so we read
#     how many characters were dropped from each row instead of guessing;
#   * DomainLength, NoOfSubDomain and CharContinuationRate come from the host
#     name; TLDLength and TLDLegitimateProb from the top-level domain;
#   * URLCharProb's exact recipe could not be recovered, so it is approximated
#     from character frequencies (correlation ~0.985 with the real values).

_SCHEME_WWW = re.compile(r"^https?://(www\.)?", re.IGNORECASE)
_URL_PARTS = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://([^/?#]*)(.*)$", re.DOTALL)
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# Count features in the order char_counts() returns them.
COUNT_COLUMNS = ["NoOfLettersInURL", "NoOfDegitsInURL", "NoOfEqualsInURL",
                 "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
                 "NoOfObfuscatedChar"]


def split_url(url):
    """'https://www.a.com/x?y=1' -> ('https', 'www.a.com', '/x?y=1'), or None."""
    m = _URL_PARTS.match(url)
    return m.groups() if m else None


def join_url(scheme, host, rest):
    return f"{scheme}://{host}{rest}"


def dropped_chars(frame):
    """How many trailing characters the dataset skipped, per row (usually 0 or 1)."""
    return (frame["URL"].astype(str).str.len() - frame["URLLength"]).clip(lower=0).astype(int)


def core_string(url, n_dropped):
    """The text the dataset counts characters in: no scheme, no 'www.'."""
    if n_dropped:
        url = url[:-n_dropped]
    return _SCHEME_WWW.sub("", url, count=1)


def char_counts(text):
    """Counts in COUNT_COLUMNS order: letters, digits, '=', '?', '&', other symbols, '%'."""
    letters = digits = symbols = 0
    for c in text:
        if c.isalpha():
            letters += 1
        elif c.isdigit():
            digits += 1
        if not c.isalnum():
            symbols += 1
    eq, qm, amp = text.count("="), text.count("?"), text.count("&")
    return letters, digits, eq, qm, amp, symbols - eq - qm - amp, text.count("%")


def domain_name(host):
    """The host without 'www.' and without the top-level domain.

    'www.my-bank.co.uk' -> 'my-bank.co'   (the dataset's convention)"""
    if host.lower().startswith("www."):
        host = host[4:]
    return host.rsplit(".", 1)[0] if "." in host else host


def _longest_run(text, is_kind):
    best = run = 0
    for c in text:
        run = run + 1 if is_kind(c) else 0
        best = max(best, run)
    return best


def char_continuation_rate(host):
    """(longest run of letters + of digits + of symbols) / length of the name.

    'ethereum-uma.com' -> name 'ethereum-uma' -> (8 + 0 + 1) / 12 = 0.75"""
    name = domain_name(host)
    if not name:
        return 0.0
    return (_longest_run(name, str.isalpha) + _longest_run(name, str.isdigit)
            + _longest_run(name, lambda c: not c.isalnum())) / len(name)


# --- Realistic edits: the moves an attacker can actually make -----------------
# Each takes (scheme, host, rest) and returns new parts, or None when the edit
# does not fit this URL (then the URL is simply left as it was).
#
# Ownership matters. An attacker can only reshape a URL whose DOMAIN they own.
# Much phishing sits on someone else's domain -- a hosting platform (web.app,
# firebaseapp.com), an IPFS gateway or link shortener (ipfs.io, bit.ly), a
# compromised organisation's site, or a bare IP address. For those we allow
# only the switch to HTTPS. A registrable domain counts as "shared" when at
# least SHARED_MIN_URLS different URLs in the training data sit on it. (That
# also catches a few attacker domains reused many times, which makes the attack
# slightly conservative -- the safe direction for a realism claim.)
SHARED_MIN_URLS = 5
_GENERIC_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "edu", "ac"}


def _editable_host(host):
    """A normal registrable name -- not an IP address, port, or 'user@host'."""
    return "." in host and not _IPV4.match(host) and not any(c in host for c in ":@[")


def _name_index(labels):
    """Position of the registrable name: 'bank' in a.bank.com and in bank.co.uk."""
    return -3 if len(labels) >= 3 and labels[-2] in _GENERIC_SECOND_LEVEL else -2


def registrable_domain(host):
    """'login.secure.bank.co.uk' -> 'bank.co.uk'   (what someone actually registers)."""
    labels = host.lower().split(".")
    return ".".join(labels[_name_index(labels):])


def shared_domains(urls, min_urls=SHARED_MIN_URLS):
    """Registrable domains that many different URLs sit on (platforms, gateways,
    shorteners, compromised sites) -- whether as subdomains or as paths."""
    distinct = set()
    for url in urls:
        parts = split_url(url)
        if parts and _editable_host(parts[1]):
            distinct.add((registrable_domain(parts[1]), url.lower()))
    per_domain = Counter(domain for domain, _ in distinct)
    return frozenset(d for d, n in per_domain.items() if n >= min_urls)


def attacker_owns_domain(host, shared):
    return _editable_host(host) and registrable_domain(host) not in shared


def edit_https(scheme, host, rest):
    """Serve the page over HTTPS. Free certificates make this cost nothing."""
    return None if scheme.lower() == "https" else ("https", host, rest)


def edit_add_www(scheme, host, rest):
    """Put the site on 'www.' -- just a subdomain of the attacker's own domain."""
    return None if host.lower().startswith("www.") else (scheme, "www." + host, rest)


def edit_clean_domain(scheme, host, rest):
    """Register a name with no digits or hyphens: 'pay-pal-24.top' -> 'paypal.top'."""
    labels = host.split(".")
    cleaned = [re.sub(r"[\d_-]", "", label) for label in labels[:-1]] + [labels[-1]]
    if any(not label for label in cleaned):
        return None                              # a whole label would vanish -- not a rename
    new_host = ".".join(cleaned)
    return None if new_host == host else (scheme, new_host, rest)


def edit_com_tld(scheme, host, rest):
    """Register the same name under .com instead of a cheap TLD (.top, .xyz, ...)."""
    labels = host.split(".")
    if labels[-1].lower() == "com":
        return None
    keep = labels[:_name_index(labels) + 1]      # drop the TLD (and a 'co.' level)
    if not keep or not keep[-1]:
        return None
    return scheme, ".".join(keep + ["com"]), rest


def edit_root_path(scheme, host, rest):
    """Serve the phishing page from the site's root: no path, no query string."""
    return None if rest in ("", "/") else (scheme, host, "")


EDITS = {
    "https": edit_https,
    "add_www": edit_add_www,
    "clean_domain": edit_clean_domain,
    "com_tld": edit_com_tld,
    "root_path": edit_root_path,
}
EDIT_LABELS = {
    "https": "serve over HTTPS",
    "add_www": "add 'www.'",
    "clean_domain": "name without digits/hyphens",
    "com_tld": "register as .com",
    "root_path": "page at site root",
}
OWNER_ONLY = {"add_www", "clean_domain", "com_tld", "root_path"}
EDIT_ORDER = ("clean_domain", "com_tld", "add_www", "root_path", "https")   # host, path, scheme


def apply_edits(url, names, shared=frozenset()):
    """Apply a set of edits to one URL. Edits that don't fit it are skipped, and
    only HTTPS is allowed on a domain the attacker doesn't own."""
    parts = split_url(url)
    if parts is None:
        return url
    owns = attacker_owns_domain(parts[1], shared)
    for name in EDIT_ORDER:
        if name in names and (owns or name not in OWNER_ONLY):
            new = EDITS[name](*parts)
            if new is not None:
                parts = new
    return join_url(*parts)


def edit_combinations(names=EDIT_ORDER):
    """Every subset of the given edits, from 'no edit' to 'all of them'."""
    return [frozenset(c) for r in range(len(names) + 1) for c in combinations(names, r)]


def combo_label(combo):
    return " + ".join(EDIT_LABELS[n] for n in EDIT_ORDER if n in combo) or "no edit"


class URLFeatureEngine:
    """Turns edited URLs back into the 21 features the model sees.

    Everything it learns (character frequencies, the URLCharProb fit, the TLD
    table) comes from the TRAINING rows only, so the attack never peeks at the
    test set."""

    def __init__(self, train_rows):
        # Domains shared by many sites: the attacker doesn't own these.
        self.shared = shared_domains(train_rows["URL"].astype(str))

        # TLDLegitimateProb is a fixed score per top-level domain (verified:
        # every TLD always has the same value), so a lookup table reproduces it.
        self.tld_prob = train_rows.groupby("TLD")["TLDLegitimateProb"].first().to_dict()

        # URLCharProb approximation: how typical the URL's characters are,
        # calibrated against the real values with a straight-line fit.
        cores = [core_string(u, k) for u, k in
                 zip(train_rows["URL"].astype(str), dropped_chars(train_rows))]
        freq = Counter(c for s in cores for c in s if c.isalnum())
        total = sum(freq.values())
        self._char_p = {c: n / total for c, n in freq.items()}
        approx = np.array([self.char_prob(s) for s in cores])
        actual = train_rows["URLCharProb"].to_numpy(dtype=float)
        self.ucp_slope, self.ucp_intercept = np.polyfit(approx, actual, 1)
        self.ucp_corr = float(np.corrcoef(approx, actual)[0, 1])

    def char_prob(self, core):
        chars = [c for c in core if c.isalnum()]
        return sum(self._char_p.get(c, 0.0) for c in chars) / len(chars) if chars else 0.0

    def recompute(self, rows, new_urls):
        """The 21 features for edited URLs (same order as `rows`).

        Every feature an edit touches is recomputed; the rest keep their
        original values. The dataset has a few counting quirks (e.g. how it
        treats '&amp;' or '%20'), and they all live in paths and query
        strings -- host names are simple, and our rules match them exactly.
        So: if the edit keeps the path, we carry the dataset's own counts
        forward and add only what changed; if the edit rewrites the path,
        nothing quirky is left and we count from scratch."""
        out = {c: rows[c].to_numpy(dtype=float).copy() for c in URL_ONLY_FEATURES}
        old_urls = rows["URL"].astype(str).to_numpy()
        dropped = dropped_chars(rows).to_numpy()

        for i, (old, new) in enumerate(zip(old_urls, new_urls)):
            old_parts, new_parts = split_url(old), split_url(new)
            if old == new or old_parts is None or new_parts is None:
                continue
            scheme_old, host_old, rest_old = old_parts
            scheme_new, host_new, rest_new = new_parts

            core_old, core_new = core_string(old, dropped[i]), core_string(new, dropped[i])
            new_counts = char_counts(core_new)
            if rest_new != rest_old:
                for col, value in zip(COUNT_COLUMNS, new_counts):
                    out[col][i] = value
            else:
                for col, before, after in zip(COUNT_COLUMNS, char_counts(core_old), new_counts):
                    out[col][i] = max(0.0, out[col][i] + after - before)
            length = max(1.0, out["URLLength"][i] + len(new) - len(old))
            out["URLLength"][i] = length

            # Ratios are re-derived from the NEW counts -- this coupling is
            # exactly what a feature-space attack gets to ignore.
            letters, digits, eq, qm, amp, other, obf = (out[c][i] for c in COUNT_COLUMNS)
            out["LetterRatioInURL"][i] = round(letters / length, 3)
            out["DegitRatioInURL"][i] = round(digits / length, 3)
            out["SpacialCharRatioInURL"][i] = round((eq + qm + amp + other) / length, 3)
            out["HasObfuscation"][i] = float(obf > 0)
            out["ObfuscationRatio"][i] = round(obf / length, 3)
            out["URLCharProb"][i] = max(0.0, out["URLCharProb"][i] + self.ucp_slope
                                        * (self.char_prob(core_new) - self.char_prob(core_old)))

            if scheme_old.lower() != scheme_new.lower():
                out["IsHTTPS"][i] = float(scheme_new.lower() == "https")
            if host_old != host_new:
                out["DomainLength"][i] += len(host_new) - len(host_old)
                out["NoOfSubDomain"][i] += host_new.count(".") - host_old.count(".")
                if domain_name(host_new) != domain_name(host_old):   # 'www.' alone doesn't count
                    out["CharContinuationRate"][i] = char_continuation_rate(host_new)
                tld_old, tld_new = host_old.rsplit(".", 1)[-1], host_new.rsplit(".", 1)[-1]
                if tld_old != tld_new:
                    out["TLDLength"][i] = len(tld_new)
                    out["TLDLegitimateProb"][i] = self.tld_prob.get(
                        tld_new, out["TLDLegitimateProb"][i])
        return pd.DataFrame(out, index=rows.index)[URL_ONLY_FEATURES]

    def variants(self, rows, combos):
        """{combo: (edited URLs, their features)} for each combination of edits."""
        urls = rows["URL"].astype(str).tolist()
        out = {}
        for combo in combos:
            if not combo:
                out[combo] = (urls, rows[URL_ONLY_FEATURES].astype(float))
                continue
            new_urls = [apply_edits(u, combo, self.shared) for u in urls]
            out[combo] = (new_urls, self.recompute(rows, new_urls))
        return out

    def owned_share(self, rows):
        """Share of these URLs whose domain the attacker owns (all moves allowed)."""
        hosts = [split_url(u)[1] if split_url(u) else "" for u in rows["URL"].astype(str)]
        return float(np.mean([attacker_owns_domain(h, self.shared) for h in hosts]))

    def check_rules(self, rows):
        """Recompute the features from scratch and compare with the dataset.

        Returns the share of rows where each reverse-engineered rule reproduces
        the stored value (to the 3 decimals the dataset stores ratios at)."""
        k = dropped_chars(rows).to_numpy()
        urls = rows["URL"].astype(str)
        hosts = rows["Domain"].astype(str)
        counts = np.array([char_counts(core_string(u, d)) for u, d in zip(urls, k)], dtype=float)
        length = rows["URLLength"].clip(lower=1).to_numpy(dtype=float)
        tld_table = rows.groupby("TLD")["TLDLegitimateProb"].first()
        mine = {
            **{col: counts[:, j] for j, col in enumerate(COUNT_COLUMNS)},
            "LetterRatioInURL": np.round(counts[:, 0] / length, 3),
            "DegitRatioInURL": np.round(counts[:, 1] / length, 3),
            "SpacialCharRatioInURL": np.round(counts[:, 2:6].sum(axis=1) / length, 3),
            "ObfuscationRatio": np.round(counts[:, 6] / length, 3),
            "HasObfuscation": (counts[:, 6] > 0).astype(float),
            "IsHTTPS": urls.str.lower().str.startswith("https://").to_numpy(dtype=float),
            "IsDomainIP": hosts.map(lambda h: bool(_IPV4.match(h))).to_numpy(dtype=float),
            "DomainLength": hosts.str.len().to_numpy(dtype=float),
            "NoOfSubDomain": (hosts.str.count(r"\.") - 1).to_numpy(dtype=float),
            "CharContinuationRate": hosts.map(char_continuation_rate).to_numpy(dtype=float),
            "TLDLength": rows["TLD"].astype(str).str.len().to_numpy(dtype=float),
            "TLDLegitimateProb": rows["TLD"].map(tld_table).to_numpy(dtype=float),
        }
        ratio_cols = {"LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL",
                      "ObfuscationRatio"}
        match = {}
        for col, vals in mine.items():
            tol = 1.1e-3 if col in ratio_cols else 1e-6
            match[col] = float(np.mean(np.abs(vals - rows[col].to_numpy(dtype=float)) <= tol))
        approx = np.array([self.char_prob(core_string(u, d)) for u, d in zip(urls, k)])
        corr = float(np.corrcoef(approx, rows["URLCharProb"].to_numpy(dtype=float))[0, 1])
        return pd.Series(match).sort_values(ascending=False), corr


def constraint_violations(features):
    """Share of rows breaking rules that every real URL's features obey.

    A feature vector that breaks one of these could not have come from any real
    URL -- which is exactly how a feature-space attack gives itself away."""
    f = features
    length = f["URLLength"].clip(lower=1)
    symbols = (f["NoOfEqualsInURL"] + f["NoOfQMarkInURL"] + f["NoOfAmpersandInURL"]
               + f["NoOfOtherSpecialCharsInURL"])
    rules = {
        "letters + digits + symbols fit inside the URL":
            f["NoOfLettersInURL"] + f["NoOfDegitsInURL"] + symbols <= f["URLLength"],
        "letter ratio matches the letter count":
            (f["LetterRatioInURL"] - f["NoOfLettersInURL"] / length).abs() <= 1.5e-3,
        "digit ratio matches the digit count":
            (f["DegitRatioInURL"] - f["NoOfDegitsInURL"] / length).abs() <= 1.5e-3,
        "symbol ratio matches the symbol count":
            (f["SpacialCharRatioInURL"] - symbols / length).abs() <= 1.5e-3,
        "every dot in the host is counted as a symbol":
            f["NoOfOtherSpecialCharsInURL"] >= f["NoOfSubDomain"],
        "the domain is shorter than the whole URL":
            f["DomainLength"] < f["URLLength"],
    }
    broken = pd.DataFrame({name: ~ok for name, ok in rules.items()})
    return broken.mean(), float(broken.any(axis=1).mean())


def raw_features(features, urls):
    """The model input for the original detector: the 21 URL features as they are."""
    return features[URL_ONLY_FEATURES]


PREFIX_DEPENDENT = ["IsHTTPS", "URLLength", "DomainLength", "NoOfSubDomain",
                    "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL",
                    "ObfuscationRatio"]


def prefix_blind(features, urls):
    """The same information, minus what an attacker changes for free.

    The scheme and a leading 'www.' cost nothing to change, so this view drops
    IsHTTPS AND measures every length and ratio WITHOUT 'http(s)://www.'.
    Deleting the IsHTTPS column alone is not enough: the dataset counts
    characters after the scheme but measures URLLength with it, so
    URLLength minus the counted characters is exactly 7, 8, 11 or 12 --
    'http://', 'https://', 'http://www.' or 'https://www.'."""
    f = features
    www = pd.Series([bool(re.match(r"^https?://www\.", u, re.IGNORECASE)) for u in urls],
                    index=f.index, dtype=float)
    symbols = (f["NoOfEqualsInURL"] + f["NoOfQMarkInURL"] + f["NoOfAmpersandInURL"]
               + f["NoOfOtherSpecialCharsInURL"])
    core = (f["NoOfLettersInURL"] + f["NoOfDegitsInURL"] + symbols).clip(lower=1)
    kept = f[[c for c in URL_ONLY_FEATURES if c not in PREFIX_DEPENDENT]]
    blind = pd.DataFrame({
        "CoreLength": core,
        "DomainLengthWithoutWWW": f["DomainLength"] - 4 * www,
        "NoOfSubDomainWithoutWWW": f["NoOfSubDomain"] - www,
        "LetterRatioOfCore": f["NoOfLettersInURL"] / core,
        "DigitRatioOfCore": f["NoOfDegitsInURL"] / core,
        "SymbolRatioOfCore": symbols / core,
        "ObfuscationRatioOfCore": f["NoOfObfuscatedChar"] / core,
    }, index=f.index)
    return pd.concat([kept, blind], axis=1)


def adaptive_attack(model, featurize, variants, combos, threshold):
    """The attacker tries every combination of edits on each URL and keeps the
    one the model finds least suspicious (like testing variants against the
    detector before launching). Returns (still_caught, chosen_combo_index, score).

    `featurize(features, urls)` turns the 21 features into this model's input."""
    scores = np.vstack([phish_proba(model, featurize(variants[c][1], variants[c][0]))
                        for c in combos])
    choice = scores.argmin(axis=0)
    best = scores[choice, np.arange(scores.shape[1])]
    return best > threshold, choice, best


# ---------------------------------------------------------------------------
# Chart style (light surface, recessive chrome, one highlight colour)
# ---------------------------------------------------------------------------
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3 (validated set)
HIGHLIGHT, DEEMPHASIS = SERIES[0], INK["muted"]
SEQUENTIAL_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                   "#256abf", "#184f95", "#0d366b"]


def use_chart_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": INK["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "legend.frameon": False,
    })


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("project_blue", SEQUENTIAL_BLUE)

r"""
Tests for common.py -- above all the URL feature engine, because the whole
attack depends on it turning an edited URL into exactly the features the
dataset would have given that URL.

Run from the project folder:
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

The "with data" tests run once the dataset has been downloaded (any script
does that on first run); otherwise they are skipped.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

ALL_MOVES = set(C.EDIT_ORDER)


class TestURLRules(unittest.TestCase):
    """The reverse-engineered rules reproduce real rows from the dataset."""

    def test_split_and_join_round_trip(self):
        for url in ["https://www.a.com/x?y=1", "http://1.2.3.4/", "https://b.co.uk"]:
            self.assertEqual(C.join_url(*C.split_url(url)), url)

    def test_real_row_dkom(self):
        # Dataset row: https://www.dkom.hr -> URLLength 18 (one character dropped),
        # 5 letters, 1 other symbol, LetterRatio 0.278, SpacialCharRatio 0.056, CCR 1.0
        url, length = "https://www.dkom.hr", 18
        core = C.core_string(url, len(url) - length)
        letters, digits, eq, qm, amp, other, pct = C.char_counts(core)
        self.assertEqual((letters, digits, other), (5, 0, 1))
        self.assertEqual(round(letters / length, 3), 0.278)
        self.assertEqual(round((eq + qm + amp + other) / length, 3), 0.056)
        self.assertEqual(C.char_continuation_rate("www.dkom.hr"), 1.0)

    def test_real_row_ethereum(self):
        # Dataset row: http://ethereum-uma.com -> URLLength 22, 13 letters,
        # 2 other symbols, LetterRatio 0.591, SpacialCharRatio 0.091, CCR 0.75
        url, length = "http://ethereum-uma.com", 22
        letters, digits, eq, qm, amp, other, pct = C.char_counts(C.core_string(url, len(url) - length))
        self.assertEqual((letters, other), (13, 2))
        self.assertEqual(round(letters / length, 3), 0.591)
        self.assertEqual(round((eq + qm + amp + other) / length, 3), 0.091)
        self.assertEqual(C.char_continuation_rate("ethereum-uma.com"), 0.75)


class TestEdits(unittest.TestCase):
    def test_each_move(self):
        self.assertEqual(C.apply_edits("http://pay.top/x", {"https"}), "https://pay.top/x")
        self.assertEqual(C.apply_edits("http://pay.top/x", {"add_www"}), "http://www.pay.top/x")
        self.assertEqual(C.apply_edits("https://pay-pal-24.top/a", {"clean_domain"}), "https://paypal.top/a")
        self.assertEqual(C.apply_edits("http://www.my-bank.co.uk/login", {"com_tld"}),
                         "http://www.my-bank.com/login")
        self.assertEqual(C.apply_edits("https://x.top/a/b?c=1", {"root_path"}), "https://x.top")

    def test_moves_that_do_not_fit_leave_the_url_alone(self):
        self.assertEqual(C.apply_edits("https://x.com/a", {"https", "com_tld"}), "https://x.com/a")
        self.assertEqual(C.apply_edits("http://www.x.com", {"add_www", "root_path"}), "http://www.x.com")
        # Cleaning '915783' would delete the whole name -- not a realistic rename.
        self.assertEqual(C.apply_edits("http://www.915783.com", {"clean_domain"}), "http://www.915783.com")

    def test_only_https_on_domains_the_attacker_does_not_own(self):
        self.assertEqual(C.apply_edits("http://198.98.58.123/login", ALL_MOVES), "https://198.98.58.123/login")
        self.assertEqual(C.apply_edits("http://abc-1.web.app/x", ALL_MOVES, frozenset({"web.app"})),
                         "https://abc-1.web.app/x")
        self.assertEqual(C.apply_edits("http://abc-1.my-site.top/x", ALL_MOVES),
                         "https://www.abc.mysite.com")

    def test_registrable_domain(self):
        self.assertEqual(C.registrable_domain("login.secure.bank.co.uk"), "bank.co.uk")
        self.assertEqual(C.registrable_domain("x.firebaseapp.com"), "firebaseapp.com")

    def test_combinations(self):
        combos = C.edit_combinations()
        self.assertEqual(len(combos), 2 ** len(C.EDIT_ORDER))
        self.assertIn(frozenset(), combos)


class TestHelpers(unittest.TestCase):
    def test_threshold_respects_false_alarm_budget(self):
        legit_scores = np.array([0.1] * 990 + [0.9] * 10)
        t = C.threshold_at_fpr(legit_scores, target_fpr=0.01)
        self.assertLessEqual(np.mean(legit_scores > t), 0.01)
        self.assertLessEqual(t, 0.1 + 1e-12)          # the lowest threshold that stays in budget

    def test_wilson_interval(self):
        lo, hi = C.wilson_interval(50, 100)
        self.assertTrue(lo < 0.5 < hi)
        self.assertAlmostEqual(hi - lo, 0.19, delta=0.01)


@unittest.skipUnless(C.CACHE_FILE.exists(), "dataset not downloaded yet -- run any script once")
class TestWithData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = C.load_data()
        cls.X, cls.y = X, y
        cls.X_train, _, cls.X_test, cls.y_train, _, cls.y_test = C.split(X, y)
        cls.engine = C.URLFeatureEngine(cls.X_train)
        phish = cls.X_test[cls.y_test.to_numpy() == 1]
        cls.sample = phish.sample(2000, random_state=C.SEED)

    def test_split_sizes_and_stratification(self):
        self.assertEqual((len(self.X_train), len(self.X_test)), (165_056, 35_370))
        self.assertAlmostEqual(self.y_train.mean(), self.y_test.mean(), places=3)

    def test_rules_reproduce_the_dataset(self):
        rules, ucp_corr = self.engine.check_rules(self.X)
        self.assertGreaterEqual(rules.min(), 0.985)
        self.assertGreaterEqual(int((rules == 1).sum()), 6)
        self.assertGreater(ucp_corr, 0.98)

    def test_no_edit_changes_nothing(self):
        variants = self.engine.variants(self.sample, [frozenset()])
        same = variants[frozenset()][1].to_numpy() == self.sample[C.URL_ONLY_FEATURES].to_numpy(float)
        self.assertTrue(same.all())

    def test_problem_space_vectors_are_consistent(self):
        variants = self.engine.variants(self.sample, C.edit_combinations())
        for combo, (_, features) in variants.items():
            _, broken = C.constraint_violations(features)
            self.assertEqual(broken, 0.0, msg=C.combo_label(combo))

    def test_prefix_blind_ignores_the_free_moves(self):
        free = [frozenset(), frozenset({"https"}), frozenset({"add_www"}), frozenset({"https", "add_www"})]
        variants = self.engine.variants(self.sample, free)
        original_urls, original_features = variants[frozenset()]
        base = C.prefix_blind(original_features, original_urls)
        for combo in free[1:]:
            urls, features = variants[combo]
            np.testing.assert_allclose(C.prefix_blind(features, urls).to_numpy(), base.to_numpy(),
                                       err_msg=C.combo_label(combo))


if __name__ == "__main__":
    unittest.main()

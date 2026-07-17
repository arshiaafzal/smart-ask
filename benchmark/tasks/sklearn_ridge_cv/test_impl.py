import unittest
from impl import RidgeCV, RidgeClassifierCV


class TestRidgeCVStoreValues(unittest.TestCase):

    def _make_regression_data(self):
        # Simple linear data: y = 2*x0 - x1
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [2.0, 0.0]]
        y = [2.0, -1.0, 1.0, 3.0, 4.0]
        return X, y

    def _make_classification_data(self):
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0],
             [0.0, 2.0], [3.0, 0.0]]
        y = ["a", "b", "a", "a", "b", "a"]
        return X, y

    def test_ridge_cv_store_cv_values(self):
        """RidgeCV must accept store_cv_values=True without TypeError."""
        X, y = self._make_regression_data()
        try:
            clf = RidgeCV(store_cv_values=True)
        except TypeError as exc:
            self.fail(f"RidgeCV raised TypeError on store_cv_values: {exc}")
        clf.fit(X, y)
        self.assertIsNotNone(clf.cv_values_)

    def test_ridge_classifier_cv_accepts_store_cv_values(self):
        """RidgeClassifierCV must accept store_cv_values=True without TypeError."""
        X, y = self._make_classification_data()
        try:
            clf = RidgeClassifierCV(store_cv_values=True)
        except TypeError as exc:
            self.fail(
                f"RidgeClassifierCV raised TypeError on store_cv_values=True: {exc}"
            )
        clf.fit(X, y)

    def test_ridge_classifier_cv_default_no_store(self):
        """Default (store_cv_values=False) must not store cv_values_."""
        X, y = self._make_classification_data()
        clf = RidgeClassifierCV()
        clf.fit(X, y)
        self.assertIsNone(clf.cv_values_)

    def test_ridge_classifier_cv_store_populates_cv_values(self):
        """When store_cv_values=True, cv_values_ must be set after fit."""
        X, y = self._make_classification_data()
        clf = RidgeClassifierCV(store_cv_values=True)
        clf.fit(X, y)
        self.assertIsNotNone(
            clf.cv_values_,
            "cv_values_ must be populated when store_cv_values=True",
        )

    def test_ridge_cv_default_no_store(self):
        """RidgeCV default must not store cv_values_."""
        X, y = self._make_regression_data()
        clf = RidgeCV()
        clf.fit(X, y)
        self.assertIsNone(clf.cv_values_)


if __name__ == "__main__":
    unittest.main()

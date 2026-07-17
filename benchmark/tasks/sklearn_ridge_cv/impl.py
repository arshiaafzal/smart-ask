"""
Minimal reproduction of scikit-learn RidgeClassifierCV store_cv_values bug.
SWE-bench: scikit-learn__scikit-learn-10297

RidgeClassifierCV does not support the `store_cv_values` parameter that
RidgeCV has, even though the docstring implies it does.  Passing
store_cv_values=True raises TypeError.

Pure-Python version (no numpy required).
"""

from __future__ import annotations


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_dot(M: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in M]


def _transpose(M: list[list[float]]) -> list[list[float]]:
    if not M:
        return []
    rows, cols = len(M), len(M[0])
    return [[M[r][c] for r in range(rows)] for c in range(cols)]


def _ridge_solve(X: list[list[float]], y: list[float], alpha: float) -> list[float]:
    """Closed-form ridge regression: coef = (X'X + alpha*I)^-1 X'y.

    Simplified: diagonal-only approximate inversion for demo purposes.
    """
    Xt = _transpose(X)
    XtX = [[_dot(Xt[i], _transpose(X)[j]) for j in range(len(Xt))]
           for i in range(len(Xt))]
    # Add alpha * I
    for i in range(len(XtX)):
        XtX[i][i] += alpha
    Xty = _mat_dot(Xt, y)
    # Simple Gaussian elimination
    n = len(XtX)
    aug = [XtX[i][:] + [Xty[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if aug[col][col] == 0:
            continue
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / aug[col][col]
            for j in range(n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] / aug[i][i] if aug[i][i] != 0 else 0.0 for i in range(n)]


class RidgeCV:
    """Simplified RidgeCV (regression) that supports store_cv_values."""

    def __init__(self, alphas=(0.1, 1.0, 10.0), store_cv_values: bool = False):
        self.alphas = list(alphas)
        self.store_cv_values = store_cv_values
        self.cv_values_: list[list[float]] | None = None
        self.alpha_: float | None = None
        self.coef_: list[float] | None = None

    def fit(self, X: list[list[float]], y: list[float]) -> "RidgeCV":
        best_alpha, best_score = None, float("inf")
        cv_vals: dict[float, list[float]] = {}
        for alpha in self.alphas:
            coef = _ridge_solve(X, y, alpha)
            preds = [_dot(row, coef) for row in X]
            mse = sum((yi - pi) ** 2 for yi, pi in zip(y, preds)) / len(y)
            if mse < best_score:
                best_score = mse
                best_alpha = alpha
            if self.store_cv_values:
                cv_vals[alpha] = preds
        self.alpha_ = best_alpha
        self.coef_ = _ridge_solve(X, y, best_alpha)
        if self.store_cv_values:
            self.cv_values_ = [cv_vals[a] for a in self.alphas]
        return self


class RidgeClassifierCV:
    """Simplified RidgeClassifierCV (classification).

    Bug: does not accept store_cv_values, raising TypeError when passed.
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        # Bug: store_cv_values parameter is missing.
    ):
        self.alphas = list(alphas)
        self.cv_values_: list | None = None
        self.alpha_: float | None = None
        self.coef_: list[float] | None = None

    def fit(self, X: list[list[float]], y: list[str]) -> "RidgeClassifierCV":
        classes = sorted(set(y))
        # Binarize: one column per class.
        Y_cols = [[1.0 if yi == c else -1.0 for yi in y] for c in classes]
        # Fit one-vs-rest ridge regression.
        best_alpha, best_score = self.alphas[0], float("inf")
        coefs = []
        for alpha in self.alphas:
            score = 0.0
            alpha_coefs = []
            for col in Y_cols:
                coef = _ridge_solve(X, col, alpha)
                preds = [_dot(row, coef) for row in X]
                score += sum((yi - pi) ** 2 for yi, pi in zip(col, preds))
                alpha_coefs.append(coef)
            if score < best_score:
                best_score = score
                best_alpha = alpha
                coefs = alpha_coefs
        self.alpha_ = best_alpha
        self.coef_ = coefs
        self.classes_ = classes
        return self

    def predict(self, X: list[list[float]]) -> list[str]:
        scores = [[_dot(row, coef) for coef in self.coef_] for row in X]
        return [self.classes_[max(range(len(s)), key=lambda i: s[i])] for s in scores]

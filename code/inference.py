"""Calendar-aware HAC covariance and tests of summed policy responses."""
import math
from statistics import NormalDist
import numpy as np

def calendar_hac_covariance(x, residuals, qdate, max_lags=4):
    """Calculate one outcome's Bartlett HAC covariance on actual quarter dates."""
    # 1. Stop on invalid bandwidths, shapes, nonfinite inputs or n <= k;
    # dropping rows here would break alignment with the fitted regression sample.
    if isinstance(max_lags, (bool, np.bool_)) or not isinstance(
        max_lags, (int, np.integer)
    ) or max_lags < 0:
        raise ValueError("max_lags must be a nonnegative integer.")
    max_lags = int(max_lags)
    x = np.asarray(x, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    qdate = np.asarray(qdate, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional design matrix.")
    n, k = x.shape
    if k == 0 or n <= k:
        raise ValueError("The design requires n > k > 0.")
    if residuals.shape != (n,) or qdate.shape != (n,):
        raise ValueError("residuals and qdate must be one-dimensional with n rows.")
    if not all(np.isfinite(a).all() for a in (x, residuals, qdate)):
        raise ValueError("Design, residuals and quarter codes must be finite.")

    # 2. Require unique, exactly representable integer quarter codes so each
    # residual pair can be assigned an unambiguous calendar lag.
    if not np.equal(qdate, np.floor(qdate)).all() or (np.abs(qdate) > 2**53).any():
        raise ValueError("qdate must contain exactly representable integer quarter codes.")
    qdate = qdate.astype(np.int64)
    if np.unique(qdate).size != n:
        raise ValueError("Each estimation quarter must be unique.")

    # 3. Require full column rank and use QR solves to transform residual-weighted
    # regressors into OLS influence rows without forming or inverting X'X.
    q, r = np.linalg.qr(x, mode="reduced")
    if np.linalg.matrix_rank(r) != k:
        raise ValueError("The design must have full column rank.")

    influence = np.linalg.solve(r, q.T).T * residuals[:, None]

    # 4. Start with zero-lag products, then add products exactly h calendar quarters
    # apart with weights 1 - h/(max_lags + 1); missing quarters never compress the lag distance.
    order = np.argsort(qdate)
    quarters = qdate[order]
    influence = influence[order]
    covariance = influence.T @ influence
    largest_distance = min(max_lags, int(quarters[-1] - quarters[0]))
    for lag in range(1, largest_distance + 1):
        earlier_quarters = quarters - lag
        earlier_rows = np.searchsorted(quarters, earlier_quarters)
        matched = quarters[earlier_rows] == earlier_quarters
        cross_product = influence[matched].T @ influence[earlier_rows[matched]]
        weight = 1 - lag / (max_lags + 1)
        covariance += weight * (cross_product + cross_product.T)

    # 5. Apply the n/(n-k) correction and remove numerical asymmetry; stop if
    # the resulting covariance is nonfinite rather than returning invalid inference.
    covariance *= n / (n - k)
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise ValueError("HAC covariance is nonfinite; inspect the numerical conditioning.")
    return covariance

def test_cumulative_response(coefficients, covariance, alpha=0.05):
    """Test whether the four policy slopes sum to zero, excluding the intercept."""
    # 1. Require coefficients ordered [intercept, b0, b1, b2, b3], their 5-by-5
    # covariance and a valid alpha; stop if dimensions or coefficient values are invalid.
    coefficients = np.asarray(coefficients, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if coefficients.shape != (5,) or covariance.shape != (5, 5):
        raise ValueError("Expected five coefficients and a 5-by-5 covariance matrix.")
    if not np.isfinite(coefficients).all():
        raise ValueError("Coefficients must be finite.")
    if not np.isfinite(alpha) or not 0 < alpha < 1 or alpha / 2 == 0:
        raise ValueError("alpha must be a representable probability strictly between 0 and 1.")

    # 2. Preserve the sum of slopes as beta and initialize inference as missing,
    # so an unusable variance cannot produce a spurious significance result.
    beta = float(coefficients[1:].sum())
    result = {
        "beta": beta,
        "se": math.nan,
        "z": math.nan,
        "pvalue": math.nan,
        "ci_lower": math.nan,
        "ci_upper": math.nan,
        "test_status": "invalid_variance",
    }

    # 3. Sum the complete slope covariance block to obtain Var(b0+b1+b2+b3),
    # including off-diagonal covariances; leave inference unavailable if nonfinite.
    slope_covariance = covariance[1:, 1:]
    if not np.isfinite(slope_covariance).all():
        return result
    variance = float(slope_covariance.sum())
    if not math.isfinite(variance):
        return result

    # 4. Set tiny negative roundoff to zero, preserving every positive variance.
    # Zero or materially negative variance returns missing inference with a status flag.
    tolerance = 32 * np.finfo(float).eps * float(np.abs(slope_covariance).sum())
    if -tolerance <= variance < 0:
        variance = 0.0
    if variance == 0:
        result["test_status"] = "zero_variance"
        return result
    if variance < 0:
        return result

    # 5. Use sqrt(variance) for the standard error and asymptotic normal critical
    # values for the two-sided p-value and (1-alpha) confidence interval.
    se = math.sqrt(variance)
    z = beta / se
    critical = -NormalDist().inv_cdf(alpha / 2)
    result.update(
        se=se,
        z=z,
        pvalue=math.erfc(abs(z) / math.sqrt(2)),
        ci_lower=beta - critical * se,
        ci_upper=beta + critical * se,
        test_status="ok",
    )
    return result

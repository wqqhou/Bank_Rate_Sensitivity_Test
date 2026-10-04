"""Legacy tables for the earlier unfiltered sample; the main report uses run_question1.py."""
from html import escape
from pathlib import Path
import pandas as pd

OUTCOMES = [("Interest income", "beta_inc"), ("Interest expense", "beta_exp")]

def _finite_values(values):
    """Exclude values that cannot be interpreted as finite numeric results."""
    # 1. Coerce invalid text and infinite values to missing before computing summary statistics.
    return pd.to_numeric(values, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )

def create_summary_table(results, alpha=0.05):
    """Give every estimated bank equal weight; use valid p-values for test shares."""
    # 1. Require a probability cutoff and the saved beta/SE/p-value columns for both core outcomes.
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    required = ["nobs"]
    for _, column in OUTCOMES:
        required.extend([column, f"{column}_se", f"{column}_pvalue"])
    missing = sorted(set(required) - set(results.columns))
    if missing:
        raise ValueError(f"Beta results are missing columns: {missing}")

    # 2. Summarize each outcome separately; finite betas enter its distribution even if its test is missing.
    records = []
    for name, column in OUTCOMES:
        beta = _finite_values(results[column])
        se = _finite_values(results[f"{column}_se"])
        pvalue = _finite_values(results[f"{column}_pvalue"])
        nobs = _finite_values(results["nobs"]).where(beta.notna())

        # 3. Use finite betas with p-values in [0, 1] as the denominator; require p < alpha to reject.
        valid_test = beta.notna() & pvalue.between(0, 1, inclusive="both")
        significant = valid_test & pvalue.lt(alpha)
        tested_banks = int(valid_test.sum())
        quantiles = beta.quantile([0.01, 0.25, 0.5, 0.75, 0.99])

        # 4. Keep effect distributions, precision and positive/negative rejections as separate summaries.
        records.append({
            "outcome": name,
            "count": int(beta.notna().sum()),
            "mean": beta.mean(),
            "std": beta.std(),
            "min": beta.min(),
            "p01": quantiles.loc[0.01],
            "p25": quantiles.loc[0.25],
            "median": quantiles.loc[0.5],
            "p75": quantiles.loc[0.75],
            "p99": quantiles.loc[0.99],
            "max": beta.max(),
            "nobs_mean": nobs.mean(),
            "nobs_median": nobs.median(),
            "tested_banks": tested_banks,
            "median_se": se.where(valid_test & se.ge(0)).median(),
            "significance_level": alpha,
            "significant_count": int(significant.sum()),
            "significant_share": (
                float(significant.sum() / tested_banks)
                if tested_banks else float("nan")
            ),
            "positive_significant_count": int((significant & beta.gt(0)).sum()),
            "negative_significant_count": int((significant & beta.lt(0)).sum()),
        })
    return pd.DataFrame(records).set_index("outcome")

def _render_summary_html(summary, min_obs=40, hac_lags=4, alpha=0.05):
    """Render the original exploratory HTML table and its method notes."""
    # 1. Check that the displayed significance threshold agrees with the supplied summary.
    if min_obs < 6 or hac_lags < 0:
        raise ValueError("Require min_obs >= 6 and hac_lags >= 0.")
    if not summary["significance_level"].eq(alpha).all():
        raise ValueError("The table and its notes must use the same alpha.")
    threshold = f"{100 * alpha:g}%"

    # 2. Format the across-bank distribution without restricting it to significant estimates.
    distribution_columns = {
        "count": "Banks", "mean": "Mean", "std": "SD",
        "min": "Min", "p25": "P25", "median": "Median",
        "p75": "P75", "max": "Max", "nobs_mean": "Mean quarters",
        "nobs_median": "Median quarters",
    }
    distribution = summary[list(distribution_columns)].rename(columns=distribution_columns)
    distribution.index.name = "Outcome"
    distribution_html = distribution.to_html(
        border=0, escape=True, na_rep="—", float_format=lambda value: f"{value:,.4f}",
        formatters={"Mean quarters": lambda value: f"{value:,.1f}",
                    "Median quarters": lambda value: f"{value:,.1f}"},
    )

    # 3. Present available-test counts and rejection shares separately from distribution statistics.
    inference_columns = {
        "tested_banks": "Valid tests", "median_se": "Median HAC SE",
        "significant_count": f"Nominal p < {alpha:g}: banks",
        "significant_share": f"Nominal p < {alpha:g}: share",
        "positive_significant_count": "Significant positive",
        "negative_significant_count": "Significant negative",
    }
    inference = summary[list(inference_columns)].rename(columns=inference_columns)
    inference.index.name = "Outcome"
    inference_html = inference.to_html(
        border=0, escape=True, na_rep="—", float_format=lambda value: f"{value:,.4f}",
        formatters={f"Nominal p < {alpha:g}: share": lambda value: f"{value:.1%}"},
    )

    # 4. Attach the earlier unfiltered sample's method notes; these are not the main report's notes.
    notes = [
        f"Each estimated bank receives equal weight. Eligibility requires at least {min_obs} "
        "shared usable quarters and a full-rank design with an intercept and four policy slopes.",
        "Beta is the sum of the contemporaneous and three lagged slopes. Its units are "
        "percentage points of annualized bank rate per one-percentage-point federal funds "
        "rate increase, cumulated over the current quarter and the next three quarters.",
        f"Tests are two-sided normal-approximation tests of cumulative beta = 0. HAC standard "
        f"errors use Bartlett weights through {hac_lags} calendar-quarter lags and the "
        "finite-sample covariance multiplier n / (n − 5).",
        f"Significance uses nominal p < {alpha:g} ({threshold}), without adjustment for multiple "
        "testing. Shares use only banks with a valid p-value; the distribution includes all "
        "finite estimated betas. Significant counts do not establish causal effects.",
        "Finite extremes and negative rates are retained; no trimming or charter restriction "
        "is applied. The CSV also includes the 1st and 99th percentiles.",
    ]
    notes_html = "\n".join(f"<li>{escape(note)}</li>" for note in notes)

    # 5. Place the two tables in a standalone, printable HTML document without editing their values.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Part 2 — Bank interest-rate betas</title>
<style>
body {{ font-family: system-ui, sans-serif; color: #1e293b; margin: 2rem auto; padding: 0 1rem; max-width: 1200px; line-height: 1.5; }}
h1 {{ font-size: 1.7rem; }} h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }}
th, td {{ padding: .6rem; border-bottom: 1px solid #dbe2ea; text-align: right; white-space: nowrap; }}
th:first-child {{ text-align: left; }} thead th {{ background: #f1f5f9; }}
.notes {{ font-size: .9rem; color: #475569; }} li {{ margin-bottom: .5rem; }}
@media print {{ body {{ max-width: none; margin: 0; }} table {{ font-size: .75rem; }} }}
</style>
</head>
<body>
<h1>Part 2 — Bank interest-rate betas</h1>
<h2>Distribution across eligible banks</h2>
<div class="table-wrap">{distribution_html}</div>
<h2>Tests of cumulative beta = 0</h2>
<div class="table-wrap">{inference_html}</div>
<ul class="notes">{notes_html}</ul>
</body>
</html>
"""


def save_summary_table(summary, output_dir, min_obs=40, hac_lags=4, alpha=0.05):
    """Save the table as CSV and standalone HTML when explicitly called."""
    # 1. Render the supplied summary with its matching threshold and HAC bandwidth.
    html = _render_summary_html(summary, min_obs=min_obs, hac_lags=hac_lags, alpha=alpha)

    # 2. Save both machine-readable statistics and the legacy HTML display in the requested folder.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "beta_summary.csv"
    html_path = output_dir / "beta_summary.html"
    summary.to_csv(csv_path, index_label="outcome")
    html_path.write_text(html, encoding="utf-8")
    return csv_path, html_path

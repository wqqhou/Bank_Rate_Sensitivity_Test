"""Question 1: bank descriptives and cross-sectional beta regressions (A/B)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _descriptives(panel):
    """Describe bank-quarter variables and the separate quarterly policy series."""
    # 1. Use positive asset levels and total assets as the equity-ratio denominator.
    # Original flows are USD thousands; divide by 1,000 for millions or 1,000,000 for billions.
    assets = panel["ave_asset"].where(panel["ave_asset"].gt(0))
    total_assets = panel["total_asset"].where(panel["total_asset"].gt(0))
    variables = {
        "Average assets (USD bn)": assets / 1e6,
        "Total assets (USD bn)": total_assets / 1e6,
        "Equity/assets (%)": 100 * panel["total_equity"] / total_assets,
        "Quarterly interest income (USD mn)": panel["interest_income_q"] / 1000,
        "Quarterly interest expense (USD mn)": panel["interest_expense_q"] / 1000,
        "Quarterly net income (USD mn)": panel["net_income_q"] / 1000,
        "Income rate (%)": panel["interest_income_rate"],
        "Expense rate (%)": panel["interest_expense_rate"],
        "ROA (%)": panel["roa_rate"],
        "Change in income rate (pp)": panel["interest_income_dif"],
        "Change in expense rate (pp)": panel["interest_expense_dif"],
        "Change in ROA (pp)": panel["roa_dif"],
    }

    # 2. Count each calendar quarter once for policy statistics, rather than
    # repeating its rate for every bank; stop if banks disagree on the quarter's values.
    policy = panel[["qdate", "FedFunds", "FedFunds_dif"]].drop_duplicates()
    if policy["qdate"].duplicated().any():
        raise ValueError("Policy values must be identical across banks in each quarter.")
    variables.update({"Fed funds (%)": policy["FedFunds"],
                      "Change in Fed funds (pp)": policy["FedFunds_dif"]})

    # 3. Compute statistics from finite observations only. Missing percentages
    # divide by each variable's observed-record count, not an expanded balanced panel.
    rows = []
    for label, series in variables.items():
        valid = series.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"Variable": label, "N": len(valid), "Mean": valid.mean(),
                     "SD": valid.std(), "P25": valid.quantile(.25),
                     "Median": valid.median(), "P75": valid.quantile(.75),
                     "Missing (%)": 100 * (1 - len(valid) / len(series))})
    return pd.DataFrame(rows).set_index("Variable")

def save_descriptives(frame, output_dir):
    """Save a statistics table, including any additional observation units."""
    # 1. Save full-precision statistics in CSV before formatting values for display.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "descriptive_stats.csv")

    # 2. Format counts as integers and other statistics to three decimals,
    # then write a LaTeX table with escaped variable labels.
    formatted = frame.copy().astype(object)
    for column in formatted:
        formatted[column] = frame[column].map(
            (lambda value: f"{value:,.0f}") if column == "N" else (lambda value: f"{value:,.3f}"))
    (output_dir / "descriptive_stats.tex").write_text(
        formatted.to_latex(escape=True, column_format="lrrrrrrr"), encoding="utf-8")

def _industry_figures(panel, output_dir, figure_dir):
    """Plot industry coverage and missingness among observed bank-quarter records."""
    # 1. Count distinct sample banks and sum positive finite total assets by quarter.
    # Divide USD thousands by 1e9 to show nominal trillions; do not impute missing assets.
    grouped = panel.groupby("qdate", sort=True)
    industry = grouped.agg(banks=("idrssd", "nunique"))
    assets = panel["total_asset"].where(np.isfinite(panel["total_asset"]) & panel["total_asset"].gt(0))
    industry["assets_trillion"] = assets.groupby(panel["qdate"]).sum(min_count=1) / 1e9
    industry.to_csv(output_dir / "industry_by_quarter.csv")
    years = 1960 + industry.index.to_numpy() / 4

    # 2. Plot bank counts and reported industry assets on their own scales,
    # using the Stata quarter codes to place observations on the calendar-year axis.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(years, industry["banks"] / 1000, color="#235789")
    axes[0].set(ylabel="Banks (thousands)", title="Commercial banks in the sample")
    axes[1].plot(years, industry["assets_trillion"], color="#28776b")
    axes[1].set(ylabel="Nominal USD trillion", title="Total reported assets")
    for ax in axes:
        ax.set(xlabel="Year", xlim=(years.min(), years.max()))
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "industry.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Flag nonfinite raw inputs, plus nonpositive average assets, separately
    # from unavailable raw annualized rates; negative finite income is not missing.
    missing = pd.DataFrame(index=panel.index)
    for column, label in [("ave_asset", "Invalid average assets"),
                          ("interest_income", "Income YTD"),
                          ("interest_expense", "Expense YTD"), ("net_income", "Net income YTD")]:
        missing[label] = ~np.isfinite(panel[column])
        if column == "ave_asset":
            missing[label] |= panel[column].le(0)
    for column, label in [("interest_income_rate_raw", "Income rate"),
                          ("interest_expense_rate_raw", "Expense rate"), ("roa_rate_raw", "ROA")]:
        missing[label] = ~np.isfinite(panel[column])

    # 4. Average the flags within each observed quarter and multiply by 100.
    # Save and plot these shares; entirely absent bank-quarters are outside the denominator.
    shares = 100 * missing.groupby(panel["qdate"]).mean()
    shares.to_csv(output_dir / "missingness_by_quarter.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, columns, title in zip(axes, [list(missing)[:4], list(missing)[4:]],
                                  ["Missing or invalid raw inputs", "Unavailable annualized rates"]):
        for column in columns:
            ax.plot(years, shares[column], label=column, linewidth=1.3)
        ax.set(xlabel="Year", ylabel="Share of observed records (%)", title=title,
               xlim=(years.min(), years.max()))
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "missingness.pdf", bbox_inches="tight")
    plt.close(fig)
    return industry, missing.mean().mul(100).to_dict()

def _cross_sections(betas):
    """Estimate income/ROA beta relationships for all banks and size subgroups."""
    # 1. Form a separate finite beta-pair sample for each dependent variable.
    # All eligible banks receive equal weight, with no additional trimming of betas.
    records, plot_data = [], []
    for outcome, label in [("beta_inc", "Income beta"), ("beta_roa", "ROA beta")]:
        frame = betas.replace([np.inf, -np.inf], np.nan).dropna(subset=["beta_exp", outcome]).copy()

        # 2. Split at this analysis's median positive mean_assets. That size
        # measure averages assets over the bank's shared income/expense estimation quarters.
        sized = frame.loc[frame["mean_assets"].gt(0) & np.isfinite(frame["mean_assets"])]
        cutoff = sized["mean_assets"].median()

        # 3. Estimate an intercept and expense-beta slope in all/small/large samples.
        # Small includes the median; HC3 standard errors and t-based inference allow heteroskedasticity.
        for sample_name, sample in [("All", frame), ("Small", sized[sized["mean_assets"].le(cutoff)]),
                                    ("Large", sized[sized["mean_assets"].gt(cutoff)])]:
            if len(sample) <= 2 or sample["beta_exp"].nunique() < 2:
                raise ValueError(f"Insufficient variation for {label}, {sample_name}.")
            x = sm.add_constant(sample[["beta_exp"]], has_constant="add")
            model = sm.OLS(sample[outcome], x).fit(cov_type="HC3", use_t=True)

            # 4. Retain coefficients, inference, sample sizes, and adjusted R2.
            # Only the all-bank model supplies the fitted line for the binned figure.
            record = {"outcome": outcome, "sample": sample_name, "nobs": int(model.nobs),
                      "adj_r2": float(model.rsquared_adj), "size_cutoff_usd_thousand": float(cutoff)}
            for term, name in [("const", "intercept"), ("beta_exp", "slope")]:
                for statistic, values in [("", model.params), ("_se", model.bse),
                                          ("_t", model.tvalues), ("_pvalue", model.pvalues)]:
                    record[name + statistic] = float(values[term])
            records.append(record)
            if sample_name == "All":
                plot_data.append((frame, outcome, label, record))
    return pd.DataFrame(records), plot_data

def _regression_table(results):
    """Arrange the six cross-sectional regressions for the report table."""
    # 1. Put t-statistics immediately below their coefficients, with bank counts
    # and adjusted R2 below; columns distinguish outcomes and size groups.
    columns, values = [], []
    for row in results.to_dict("records"):
        label = "Income beta" if row["outcome"] == "beta_inc" else "ROA beta"
        columns.append((label, row["sample"]))
        values.append([f"{row['intercept']:.3f}", f"({row['intercept_t']:.2f})",
                       f"{row['slope']:.3f}", f"({row['slope_t']:.2f})",
                       f"{row['nobs']:,}", f"{row['adj_r2']:.3f}"])
    return pd.DataFrame(np.array(values).T, columns=pd.MultiIndex.from_tuples(columns),
                        index=["Intercept", "", "Expense beta", "", "Banks", r"Adjusted $R^2$"])

def _relationship_figure(plot_data, output_dir, figure_dir):
    """Plot quantile-bin averages alongside OLS fitted to individual banks."""
    # 1. Allocate one panel to each all-bank relationship, sharing the same
    # plotting conventions for income and profitability sensitivities.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (frame, outcome, label, model) in zip(axes, plot_data):

        # 2. Form up to 20 expense-beta quantile bins; ties can reduce the count.
        # Each plotted point is the unweighted mean beta pair of its constituent banks.
        frame["bin"] = pd.qcut(frame["beta_exp"], q=min(20, len(frame)), duplicates="drop")
        bins = frame.groupby("bin", observed=True).agg(
            expense_beta=("beta_exp", "mean"), outcome_beta=(outcome, "mean"), banks=(outcome, "size"))
        bins.to_csv(output_dir / f"bins_{outcome}.csv", index=False)

        # 3. Overlay the OLS line estimated on every eligible individual bank,
        # rather than estimating a new regression on the displayed bin means.
        ax.scatter(bins["expense_beta"], bins["outcome_beta"], s=26, color="#235789", label="Bin means")
        grid = np.linspace(bins["expense_beta"].min(), bins["expense_beta"].max(), 100)
        ax.plot(grid, model["intercept"] + model["slope"] * grid, color="#c55a32",
                label="OLS on individual banks")
        ax.set(xlabel="Expense beta", ylabel=label, title=f"{label} versus expense beta")
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)

    # 4. Save both relationships together after writing the underlying bin data.
    fig.tight_layout()
    fig.savefig(figure_dir / "beta_relationships.pdf", bbox_inches="tight")
    plt.close(fig)

def run_bank_analysis(panel, betas, output_dir, figure_dir):
    """Write Part 1 and Part 3A/B outputs and return report-ready numbers."""
    # 1. Prepare output directories and reject duplicate bank-quarter or bank-beta
    # keys before counts, descriptive statistics, and cross-sectional regressions.
    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if panel.duplicated(["idrssd", "qdate"]).any() or betas["idrssd"].duplicated().any():
        raise ValueError("Panel or bank-beta identifiers are duplicated.")

    # 2. Build the descriptive table, industry/missingness figures, and beta
    # relationships from the supplied cleaned panel and estimated bank-level betas.
    descriptive = _descriptives(panel)
    save_descriptives(descriptive, output_dir)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        industry, missing = _industry_figures(panel, output_dir, figure_dir)
        results, plot_data = _cross_sections(betas)
        _relationship_figure(plot_data, output_dir, figure_dir)

    # 3. Save full regression statistics in CSV and the coefficient/t-statistic
    # presentation in LaTeX, keeping display formatting separate from numeric results.
    results.to_csv(output_dir / "cross_section_results.csv", index=False)
    table = _regression_table(results)
    (output_dir / "cross_section_results.tex").write_text(
        table.to_latex(escape=False, multicolumn_format="c", column_format="lrrrrrr"), encoding="utf-8")

    # 4. Record coverage, missingness denominators, and regression choices so
    # the report can reuse computed quantities without manually transcribing them.
    summary = {
        "n_bank_quarters": len(panel), "n_banks": int(panel["idrssd"].nunique()),
        "industry_first": {"qdate": int(industry.index[0]), **industry.iloc[0].to_dict()},
        "industry_last": {"qdate": int(industry.index[-1]), **industry.iloc[-1].to_dict()},
        "missing_percent": missing, "cross_sections": results.to_dict("records"),
        "methods": {"standard_errors": "HC3; t statistics use coefficients divided by HC3 SEs",
                    "size": "Within-analysis median positive mean assets; small <= median, large > median",
                    "bins": "Up to 20 expense-beta quantile bins; ties may reduce their number; no beta trimming",
                    "industry": "Commercial-bank count and sum of positive reported nominal total assets",
                    "missingness": "Shares among observed records; absent bank-quarters are not counted",
                    "policy_descriptives": "One observation per calendar quarter, not repeated across banks"},
    }
    (output_dir / "bank_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

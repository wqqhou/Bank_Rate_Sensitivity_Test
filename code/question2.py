"""Question 2: relate commercial banks' cash-share changes to policy-rate changes."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fmc-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from panel import PROJECT_ROOT, f_construction
from estimation import REGRESSORS
from inference import calendar_hac_covariance, test_cumulative_response

OUTPUT = PROJECT_ROOT / "data/question2"
REPORT = PROJECT_ROOT / "report"
MIN_OBS, HAC_LAGS = 40, 4


def construct_panel():
    """Construct cash shares independently of Question 1's income sample."""
    # 1. Read balance-sheet stocks and require the same valid, unique bank-quarter
    # keys as Question 1; structural errors stop the run instead of deleting records.
    columns = ["idrssd", "year", "qdate", "charter_type", "cash", "total_asset"]
    panel = pd.read_stata(PROJECT_ROOT / "data/raw/callreports_1984_2017.dta",
                          columns=columns, convert_dates=False).sort_values(["idrssd", "qdate"])
    assert panel[["idrssd", "qdate"]].notna().all().all(), "Missing bank/date keys"
    assert panel.idrssd.gt(0).all() and panel.idrssd.mod(1).eq(0).all(), "Invalid bank ID"
    assert not panel.duplicated(["idrssd", "qdate"]).any(), "Duplicate bank-quarter"
    assert panel.qdate.mod(1).eq(0).all(), "Noninteger quarter code"
    assert panel.year.eq(1960 + panel.qdate.floordiv(4)).all(), "Year/quarter mismatch"
    assert panel.qdate.between(96, 231).all(), "Quarter outside 1984Q1--2017Q4"
    audit = {"RawRows": len(panel)}

    # 2. Keep commercial-bank quarters, then mask shares unless assets are finite
    # and positive and cash is finite between zero and assets, including both endpoints.
    panel = panel.loc[panel.charter_type.eq(200)].reset_index(drop=True).copy()
    invalid_assets = ~np.isfinite(panel.total_asset) | panel.total_asset.le(0)
    nonfinite_cash = ~np.isfinite(panel.cash)
    negative_cash = np.isfinite(panel.cash) & panel.cash.lt(0)
    above_assets = np.isfinite(panel.cash) & np.isfinite(panel.total_asset) & panel.cash.gt(panel.total_asset)
    valid = ~(invalid_assets | nonfinite_cash | negative_cash | above_assets)
    panel["cash_share"] = 100 * panel.cash.where(valid) / panel.total_asset.where(valid)
    audit.update(PanelRows=len(panel), PanelBanks=int(panel.idrssd.nunique()),
                 OtherCharterRows=audit["RawRows"] - len(panel),
                 InvalidAssetRows=int(invalid_assets.sum()), NonfiniteCashRows=int(nonfinite_cash.sum()),
                 NegativeCashRows=int(negative_cash.sum()), CashAboveAssetsRows=int(above_assets.sum()),
                 InvalidRatioRows=int((~valid).sum()), ValidRatioRows=int(valid.sum()))

    # 3. These are stocks, so use no YTD conversion, annualization or winsorization.
    # Difference only adjacent commercial-bank quarters; decreases remain valid.
    adjacent = panel.groupby("idrssd").qdate.diff().eq(1)
    panel["cash_share_dif"] = panel.groupby("idrssd").cash_share.diff().where(adjacent)
    audit["UnavailableDiffRows"] = int(panel.cash_share_dif.isna().sum())

    # 4. Reuse Question 1's quarter-end-month policy rates and calendar lags;
    # require exactly one policy row per quarter and a match for every bank record.
    panel = panel.merge(f_construction(), on="qdate", how="left", validate="many_to_one", indicator=True)
    assert panel._merge.eq("both").all(), "Bank quarter has no policy observation"
    panel = panel.drop(columns="_merge")
    audit["UsableRows"] = int(np.isfinite(panel[["cash_share_dif"] + REGRESSORS]).all(axis=1).sum())
    return panel, audit


def estimate_banks(panel):
    """Estimate one cash-share model per bank and audit every bank's eligibility."""
    # 1. Keep complete finite regression rows within each bank; 40 usable quarters
    # need not be consecutive, and no income, expense or ROA observations are required.
    records, audits = [], []
    for bank_id, bank in panel.groupby("idrssd", sort=True):
        sample = bank.loc[np.isfinite(bank[["cash_share_dif"] + REGRESSORS]).all(axis=1)]
        n = len(sample)
        audit = {"idrssd": int(bank_id), "n_input_rows": len(bank), "nobs": n,
                 "rank": None, "status": "too_few_observations"}
        if n < MIN_OBS:
            audits.append(audit)
            continue

        # 2. Fit an intercept and four policy slopes; all five parameters must
        # be identified. Keep every successful estimate regardless of significance.
        x = np.column_stack([np.ones(n), sample[REGRESSORS].to_numpy(dtype=float)])
        y = sample.cash_share_dif.to_numpy(dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        audit.update(rank=int(rank), status="estimated" if rank == 5 else "rank_deficient")
        audits.append(audit)
        if rank < 5:
            continue
        residuals = y - x @ coefficients

        # 3. Calculate calendar-quarter HAC4 covariance and test the sum of four
        # slopes. Constant outcomes retain a beta but have no usable fit or inference.
        constant = bool(np.all(y == y[0]))
        covariance = (np.zeros((5, 5)) if constant else
                      calendar_hac_covariance(x, residuals, sample.qdate.to_numpy(), HAC_LAGS))
        test = test_cumulative_response(coefficients, covariance)
        if constant:
            test["test_status"] = "constant_outcome"
        r2 = np.nan if constant else 1 - float(residuals @ residuals) / float((y - y.mean()) @ (y - y.mean()))
        record = {"idrssd": int(bank_id), "nobs": n, "qdate_first": int(sample.qdate.min()),
                  "qdate_last": int(sample.qdate.max()), "intercept": float(coefficients[0]),
                  **{f"b{lag}": float(coefficients[lag + 1]) for lag in range(4)},
                  "beta_cash": test["beta"], "r_squared": r2,
                  "adjusted_r_squared": 1 - (1 - r2) * (n - 1) / (n - 5)}
        record.update({f"beta_cash_{key}": value for key, value in test.items() if key != "beta"})
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(audits)


def save_summary(banks):
    """Summarize bank estimates without filtering on statistical significance."""
    # 1. Weight banks equally. Inference summaries require status='ok', a finite
    # positive SE and a p-value in [0,1]; unavailable tests stay in beta summaries.
    se, p = banks.beta_cash_se, banks.beta_cash_pvalue
    valid_test = banks.beta_cash_test_status.eq("ok") & np.isfinite(se) & se.gt(0) & p.between(0, 1)
    summary = {"EstimatedBanks": len(banks), "EstimationRows": int(banks.nobs.sum()),
               "ValidTests": int(valid_test.sum()), "UnavailableTests": int((~valid_test).sum()),
               "SignificantBanks": int((valid_test & p.lt(.05)).sum()),
               "UnavailableFitStatistics": int(banks.r_squared.isna().sum()),
               "MeanBeta": float(banks.beta_cash.mean()), "MedianBeta": float(banks.beta_cash.median()),
               "MedianSE": float(se.loc[valid_test].median()),
               "SignificantShare": float(100 * p.loc[valid_test].lt(.05).mean()),
               "MedianRSquared": float(banks.r_squared.median()),
               "MedianAdjustedRSquared": float(banks.adjusted_r_squared.median())}

    # 2. Save unrounded values and a compact report table. R-squared values are
    # fractions, not percentages; negative adjusted R-squared values remain unchanged.
    labels = {"EstimatedBanks": "Eligible banks", "MeanBeta": "Mean cumulative beta",
              "MedianBeta": "Median cumulative beta", "MedianSE": "Median HAC standard error",
              "SignificantShare": r"Banks with cumulative $p<0.05$ (\%)",
              "MedianRSquared": r"Median $R^2$", "MedianAdjustedRSquared": r"Median adjusted $R^2$"}
    table = pd.DataFrame({"Metric": list(labels.values()), "Value": [summary[key] for key in labels]})
    table.to_csv(OUTPUT / "cash_summary.csv", index=False)
    table["Value"] = [f"{summary[key]:,.0f}" if key == "EstimatedBanks" else
                      f"{summary[key]:.1f}" if key == "SignificantShare" else f"{summary[key]:.3f}"
                      for key in labels]
    (OUTPUT / "cash_summary.tex").write_text(
        table.to_latex(index=False, escape=False, column_format="lr"), encoding="utf-8")
    return summary


def save_figure(panel, banks):
    """Plot eligible banks' mean cash shares and the common policy-rate series."""
    # 1. Average all available valid level observations of banks eligible for the
    # regression, including their first quarters; the number of banks can change over time.
    eligible = panel.loc[panel.idrssd.isin(banks.idrssd)]
    levels = eligible.groupby("qdate").agg(mean_cash_share=("cash_share", "mean"),
                                          n_banks=("cash_share", "count"))
    series = f_construction()[["qdate", "FedFunds"]].merge(levels, on="qdate", how="left", validate="one_to_one")
    series.insert(1, "quarter", pd.period_range("1984Q1", "2017Q4", freq="Q").astype(str))
    series.to_csv(OUTPUT / "cash_by_quarter.csv", index=False)

    # 2. Use separate, aligned level panels; the regression instead uses changes.
    # Equal bank weights and changing coverage are stated in the figure and saved counts.
    years = 1960 + series.qdate.to_numpy() / 4
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.7), sharex=True)
        axes[0].plot(years, series.mean_cash_share, color="#235789")
        axes[0].set(ylabel="Cash and bank balances/assets (%)",
                    title="Eligible banks: equal-bank mean; coverage changes over time")
        axes[1].plot(years, series.FedFunds, color="#28776b")
        axes[1].set(ylabel="Federal funds rate (%)", xlabel="Year", title="Final month of each quarter")
        for ax in axes:
            ax.set_xlim(years.min(), years.max())
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(REPORT / "figures/question2_cash.pdf", bbox_inches="tight")
        plt.close(fig)
    return {"FirstMeanCash": float(series.mean_cash_share.iloc[0]),
            "LastMeanCash": float(series.mean_cash_share.iloc[-1])}


def main():
    """Reproduce Question 2 directly from the two supplied raw datasets."""
    # 1. Save the constructed panel and all bank eligibility/estimation records;
    # leave Question 1's inputs, outputs and scripts untouched.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    panel, audit = construct_panel()
    panel.to_csv(OUTPUT / "cash_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    banks, eligibility = estimate_banks(panel)
    banks.to_csv(OUTPUT / "cash_betas.csv", index=False)
    eligibility.to_csv(OUTPUT / "cash_eligibility.csv", index=False)
    audit.update(TooFewBanks=int(eligibility.status.eq("too_few_observations").sum()),
                 RankDeficientBanks=int(eligibility.status.eq("rank_deficient").sum()))

    # 2. Generate the compact evidence and method metadata; do not claim a causal
    # effect or calculate a test of the average beta across correlated banks.
    summary = {**save_summary(banks), **save_figure(panel, banks)}
    methods = {"min_observations": MIN_OBS, "hac_lags": HAC_LAGS,
               "cash_definition": "Cash and balances due from depository institutions, not Fed reserves alone",
               "sample": "Independent of Question 1 income/expense/ROA availability; charter_type=200",
               "ratio": "100*cash/total_asset; finite assets>0 and finite 0<=cash<=assets",
               "stock_treatment": "No YTD conversion, annualization or winsorization; no clipping",
               "differences": "Adjacent calendar quarters only; negative changes retained",
               "covariance": "Calendar-quarter Bartlett HAC4 with n/(n-5) correction",
               "test": "Sum of four slopes=0; two-sided asymptotic normal; nominal alpha=.05; no multiplicity adjustment",
               "fit_statistics": "In-sample R-squared and adjusted R-squared; constant outcomes unavailable",
               "figure": "Equal-bank mean among eligible banks' available level records; composition varies by quarter"}
    (OUTPUT / "results.json").write_text(json.dumps(
        {"data_audit": audit, "summary": summary, "methods": methods}, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # 3. Export rounded report values directly from results; counts remain integers,
    # significance shares use one decimal, and rates/betas/fit statistics use three.
    decimals = {key: (1 if key == "SignificantShare" else 3) for key in
                ["MeanBeta", "MedianBeta", "MedianSE", "SignificantShare", "MedianRSquared",
                 "MedianAdjustedRSquared", "FirstMeanCash", "LastMeanCash"]}
    numbers = {key: f"{value:.{decimals[key]}f}" if key in decimals else f"{value:,}"
               for key, value in {**audit, **summary}.items()}
    (REPORT / "question2_numbers.tex").write_text("% Generated by code/question2.py\n" + "\n".join(
        rf"\newcommand{{\QTwo{key}}}{{{value}}}" for key, value in numbers.items()) + "\n", encoding="utf-8")
    print(json.dumps({"data_audit": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

"""Reproduce Question 1 tables, figures, estimates, and LaTeX report numbers."""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fmc-matplotlib"))
import pandas as pd
import statsmodels.api as sm
from panel import PROJECT_ROOT, RATE_NAMES, panel_construction, add_differences
from estimation import estimate_bank_betas
from bank_analysis import run_bank_analysis, save_descriptives
from market import run_market

OUTPUT = PROJECT_ROOT / "data/question1"
REPORT = PROJECT_ROOT / "report"


def beta_summary(betas):
    rows = {}
    for name, label in [("inc", "Income beta"), ("exp", "Expense beta"), ("roa", "ROA beta")]:
        s = betas[f"beta_{name}"].dropna()
        rows[label] = {"Banks": len(s), "Mean": s.mean(), "SD": s.std(),
                       "P25": s.quantile(.25), "Median": s.median(), "P75": s.quantile(.75)}
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.to_csv(OUTPUT / "beta_summary.csv")
    formatted = summary.map(lambda value: f"{value:.3f}")
    formatted["Banks"] = summary.Banks.map(lambda value: f"{value:,.0f}")
    (OUTPUT / "beta_summary.tex").write_text(formatted.to_latex(column_format="lrrrrrr"), encoding="utf-8")
    return rows


def add_market_descriptives():
    table = pd.read_csv(OUTPUT / "descriptive_stats.csv", index_col="Variable")
    market = pd.read_csv(OUTPUT / "company_month_panel.csv.gz", usecols=["return", "log_assets"])
    for label, values in [("Monthly stock return (pp)", 100 * market["return"]),
                          ("Log company assets (USD thousand)", market.log_assets)]:
        table.loc[label] = [values.count(), values.mean(), values.std(), values.quantile(.25),
                            values.median(), values.quantile(.75), 100 * values.isna().mean()]
    save_descriptives(table, OUTPUT)


def sensitivity(panel):
    # One check isolates how winsorization changes the two requested cross-sectional slopes.
    raw = panel.copy()
    for name in RATE_NAMES:
        raw[f"{name}_rate"] = raw[f"{name}_rate_raw"]
    estimates, _ = estimate_bank_betas(add_differences(raw))
    result = {}
    for name in ["inc", "roa"]:
        sample = estimates.dropna(subset=[f"beta_{name}", "beta_exp"])
        fit = sm.OLS(sample[f"beta_{name}"], sm.add_constant(sample.beta_exp)).fit(cov_type="HC3")
        result[name] = {"slope": float(fit.params.beta_exp), "t": float(fit.tvalues.beta_exp),
                        "nobs": int(fit.nobs)}
    return result


def report_numbers(audit, eligibility, summary, banks, market, robustness):
    numbers = {
        "RawRows": f"{audit['raw_rows']:,}", "RawBanks": f"{audit['raw_banks']:,}",
        "PanelRows": f"{audit['commercial_rows']:,}", "PanelBanks": f"{audit['commercial_banks']:,}",
        "OtherCharters": f"{audit['other_charter_rows']:,}", "Gaps": f"{audit['internal_quarter_gaps']:,}",
        "BanksFirst": f"{banks['industry_first']['banks']:,.0f}",
        "BanksLast": f"{banks['industry_last']['banks']:,.0f}",
        "AssetsFirst": f"{banks['industry_first']['assets_trillion']:.2f}",
        "AssetsLast": f"{banks['industry_last']['assets_trillion']:.2f}",
        "CoreEligible": f"{eligibility.status.eq('estimated').sum():,}",
        "CoreExcluded": f"{eligibility.status.ne('estimated').sum():,}",
        "RoaEligible": f"{eligibility.status_roa.eq('estimated').sum():,}",
        "LinkedBanks": f"{market['matched_banks_all_years']:,}",
        "LinkedCompanies": f"{market['matched_companies_all_years']:,}",
        "MarketBanks": f"{market['matched_banks_in_stock_analysis']:,}",
        "MarketCompanies": f"{market['matched_companies_in_stock_analysis']:,}",
        "MarketObservations": f"{market['regression_nobs']:,}",
        "IncompleteGroups": f"{market['incomplete_company_quarters_excluded']:,}",
        "StockDuplicates": f"{market['duplicate_security_month_rows_removed']:,}",
        "MarketLastIndex": f"{market['index_last_value']:.1f}",
    }
    for name, label in [("Inc", "Income beta"), ("Exp", "Expense beta"), ("Roa", "ROA beta")]:
        for stat in ["Mean", "Median"]:
            numbers[f"{name}{stat}"] = f"{summary[label][stat]:.3f}"
    for row in banks["cross_sections"]:
        prefix = ("Inc" if row["outcome"] == "beta_inc" else "Roa") + row["sample"]
        for key, suffix in [("slope", "Slope"), ("intercept", "Intercept"), ("adj_r2", "Rsq")]:
            numbers[prefix + suffix] = f"{row[key]:.3f}"
    numbers["SizeCutoff"] = f"{banks['cross_sections'][0]['size_cutoff_usd_thousand'] / 1000:.1f}"
    for name, values in robustness.items():
        numbers[f"Raw{name.title()}Slope"] = f"{values['slope']:.3f}"
        numbers[f"Raw{name.title()}T"] = f"{values['t']:.2f}"
    table = pd.read_csv(OUTPUT / "stock_return_regression.csv", index_col="variable")
    labels = {"log_assets": "Log assets", "roa": "Annualized ROA (pp)", "capital": "Equity/assets (pp)"}
    lines = [r"\begin{tabular}{lr}", r"\toprule", r" & Monthly return (pp) \\", r"\midrule"]
    for variable, label in labels.items():
        row = table.loc[variable]
        lines.extend([f"{label} & {row.coefficient:.3f} " + r"\\",
                      f" & ({row.t_stat:.2f}) " + r"\\"])
        numbers[{"log_assets": "StockSize", "roa": "StockRoa", "capital": "StockCapital"}[variable]] = f"{row.coefficient:.3f}"
    lines.extend([r"Company and month effects & Yes \\",
                  f"Company-months & {market['regression_nobs']:,} " + r"\\",
                  f"Companies & {market['regression_companies']:,} " + r"\\",
                  f"Adjusted $R^2$ (including effects) & {market['regression_adjusted_r2']:.3f} " + r"\\",
                  r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "stock_return_regression.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT / "numbers.tex").write_text("% Generated by code/run_question1.py\n" + "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in numbers.items()) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    print("1/5: Constructing and saving the bank panel", flush=True)
    panel = panel_construction()
    audit = panel.attrs["audit"]
    panel.to_csv(OUTPUT / "bank_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    print("2/5: Estimating bank income, expense, and ROA betas", flush=True)
    betas, eligibility = estimate_bank_betas(panel)
    betas.dropna(subset=["beta_inc", "beta_exp"]).to_csv(OUTPUT / "bank_betas.csv", index=False)
    betas[["idrssd", "beta_roa", "nobs_roa"]].dropna().to_csv(OUTPUT / "bank_roa_betas.csv", index=False)
    eligibility.to_csv(OUTPUT / "bank_eligibility.csv", index=False)
    summary = beta_summary(betas)
    banks = run_bank_analysis(panel, betas, OUTPUT, REPORT / "figures")
    print("3/5: Linking listed companies and estimating the return regression", flush=True)
    market = run_market(panel, OUTPUT, REPORT / "figures")
    add_market_descriptives()
    print("4/5: Checking sensitivity to extreme rate observations", flush=True)
    robustness = sensitivity(panel)
    print("5/5: Writing reproducible report numbers", flush=True)
    report_numbers(audit, eligibility, summary, banks, market, robustness)
    results = {"data_audit": audit, "beta_summary": summary, "bank_analysis": banks,
               "market": market, "without_winsorization": robustness,
               "reported_time_minutes": {"part1": 20, "part2": 90, "part3": 90}}
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Complete: {OUTPUT}\nCompile: latexmk -pdf -cd report/main.tex", flush=True)


if __name__ == "__main__":
    main()

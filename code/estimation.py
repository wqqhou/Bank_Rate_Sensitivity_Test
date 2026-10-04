"""Separate bank OLS regressions; sum the contemporaneous and three lag slopes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from panel import PROJECT_ROOT, panel_construction

REGRESSORS = ["FedFunds_dif"] + [f"FedFunds_dif_lag{k}" for k in (1, 2, 3)]
OUTCOMES = ["interest_income_dif", "interest_expense_dif"]


def fit_bank(sample, outcomes, names, min_obs, inference=False, hac_lags=4):
    """Fit the requested outcomes on one bank's shared usable quarters."""
    # 1. Drop rows from this fit if a requested outcome or policy regressor is nonfinite;
    # skip this fit with an audit reason if fewer than min_obs quarters remain.
    sample = sample.loc[np.isfinite(sample[outcomes + REGRESSORS]).all(axis=1)]
    if len(sample) < min_obs:
        return {}, len(sample), "too_few_observations"

    # 2. Estimate an intercept and four policy slopes on the same design for
    # every requested outcome; skip designs that cannot identify all five parameters.
    x = np.column_stack([np.ones(len(sample)), sample[REGRESSORS].to_numpy(dtype=float)])
    y = sample[outcomes].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        return {}, len(sample), "rank_deficient"

    # 3. Store each outcome's cumulative response by summing the contemporaneous
    # and three lag coefficients, leaving the intercept out of the beta.
    record = {}
    for j, name in enumerate(names):
        record[f"beta_{name}"] = float(coefficients[1:, j].sum())

        # 4. If requested, test the slope sum using its full calendar-aware HAC
        # covariance; flag constant outcomes and leave their inference unavailable.
        if inference:
            from inference import calendar_hac_covariance, test_cumulative_response
            covariance = calendar_hac_covariance(x, y[:, j] - x @ coefficients[:, j],
                                                 sample.qdate.to_numpy(), hac_lags)
            test = test_cumulative_response(coefficients[:, j], covariance)
            if np.all(y[:, j] == y[0, j]):
                test = test_cumulative_response(coefficients[:, j], np.zeros((5, 5)))
                test["test_status"] = "constant_outcome"
            record.update({f"beta_{name}_{key}": value for key, value in test.items() if key != "beta"})

    # 5. Measure bank size and date coverage on the income/expense estimation
    # sample so later size comparisons use the same observations as those betas.
    if "inc" in names:
        record.update(mean_assets=float(sample.ave_asset.mean()),
                      qdate_first=int(sample.qdate.min()), qdate_last=int(sample.qdate.max()))
    return record, len(sample), "estimated"


def estimate_bank_betas(panel, min_obs=40, hac_lags=4, inference=False):
    """Estimate income/expense on common quarters and ROA on its own sample."""
    # 1. Default to 40 usable quarters per fit; reject minima below six because
    # five parameters need residual degrees of freedom, and reject invalid HAC bandwidths.
    if min_obs < 6:
        raise ValueError("At least six observations are needed for five parameters.")
    if not isinstance(hac_lags, (int, np.integer)) or hac_lags < 0:
        raise ValueError("HAC lags must be a nonnegative integer.")

    # 2. Require the model fields and unique, nonmissing bank-quarter keys;
    # structural input errors stop the run rather than silently altering the sample.
    required = ["idrssd", "qdate", "ave_asset", "roa_dif"] + OUTCOMES + REGRESSORS
    if set(required) - set(panel.columns):
        raise ValueError(f"Missing columns: {set(required) - set(panel.columns)}")
    if panel[["idrssd", "qdate"]].isna().any().any() or panel.duplicated(["idrssd", "qdate"]).any():
        raise ValueError("Bank-quarter keys must be present and unique.")

    # 3. Process each bank in calendar order, sharing complete cases between
    # income and expense while allowing ROA to use its own available quarters.
    records, audits = [], []
    for idrssd, bank in panel.sort_values(["idrssd", "qdate"]).groupby("idrssd", sort=True):
        core, nobs, status = fit_bank(bank, OUTCOMES, ["inc", "exp"], min_obs, inference, hac_lags)
        roa, nobs_roa, status_roa = fit_bank(bank, ["roa_dif"], ["roa"], min_obs, inference, hac_lags)
        counts = {"idrssd": int(idrssd), "nobs": nobs, "nobs_roa": nobs_roa}

        # 4. Audit every bank, including rejected fits, and keep an estimate row
        # whenever either the income/expense model or the ROA model succeeds.
        audits.append({**counts, "n_input_rows": len(bank), "status": status, "status_roa": status_roa})
        if core or roa:
            records.append({**counts, **core, **roa})
    return pd.DataFrame(records), pd.DataFrame(audits)


def save_bank_estimates(betas, audit, output_dir):
    """Export eligible income/expense betas, ROA betas and the full audit."""
    # 1. Create the destination and export banks with both core betas, retaining
    # any other available columns, including optional inference and sample details.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    betas.dropna(subset=["beta_inc", "beta_exp"]).to_csv(output_dir / "bank_betas.csv", index=False)

    # 2. Export ROA-eligible banks separately, retaining missing inference when
    # the point estimate exists; the audit CSV includes all processed banks.
    roa_columns = ["idrssd", "nobs_roa"] + [c for c in betas if c.startswith("beta_roa")]
    betas[roa_columns].dropna(subset=["beta_roa"]).to_csv(output_dir / "bank_roa_betas.csv", index=False)
    audit.to_csv(output_dir / "bank_eligibility.csv", index=False)


def main():
    """Run bank estimation from raw files or a supplied panel CSV."""
    # 1. Read the input/output choices and estimation settings, with 40 usable
    # quarters required by default and optional HAC inference using four calendar lags.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--min-obs", type=int, default=40)
    parser.add_argument("--inference", action="store_true", help="Also compute optional cumulative-response HAC tests.")
    parser.add_argument("--hac-lags", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/question1")
    args = parser.parse_args()

    # 2. Load the supplied panel or construct it from raw data, then estimate
    # the requested models and save eligible-bank files plus their audit records.
    panel = pd.read_csv(args.panel) if args.panel else panel_construction()
    betas, audit = estimate_bank_betas(panel, args.min_obs, args.hac_lags, args.inference)
    save_bank_estimates(betas, audit, args.output_dir)
    print(f"Saved estimates and eligibility records to {args.output_dir}")


if __name__ == "__main__":
    main()

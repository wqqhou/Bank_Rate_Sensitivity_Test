"""Build Question 1's quarterly commercial-bank panel from the supplied files."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE_NAMES = ["interest_income", "interest_expense", "roa"]

def add_differences(df):
    """Differences require the immediately preceding calendar quarter."""
    # 1. Subtract the previous rate within each bank only when quarter codes
    # differ by one; this keeps Q4-to-Q1 changes and masks changes across gaps.
    adjacent = df.groupby("idrssd")["qdate"].diff().eq(1)
    for name in RATE_NAMES:
        df[f"{name}_dif"] = df.groupby("idrssd")[f"{name}_rate"].diff().where(adjacent)
    return df

def y_construction(winsorize=True):
    """Construct quarterly flows and cleaned bank outcome rates."""
    # 1. Read the required bank fields and order observations by bank and quarter.
    # Preserve numeric Stata quarter codes so a one-unit change means one quarter.
    columns = ["idrssd", "year", "qdate", "charter_type", "bhc_id", "total_asset",
               "total_equity", "ave_asset", "interest_income", "interest_expense", "net_income"]
    df = pd.read_stata(PROJECT_ROOT / "data/raw/callreports_1984_2017.dta",
                      columns=columns, convert_dates=False).sort_values(["idrssd", "qdate"])

    # 2. Stop on missing/invalid keys, duplicate bank-quarters, inconsistent years,
    # or dates outside 1984Q1--2017Q4; these checks do not silently delete records.
    assert df[["idrssd", "qdate"]].notna().all().all(), "Missing bank/date keys"
    assert df.idrssd.gt(0).all() and df.idrssd.mod(1).eq(0).all(), "Invalid bank ID"
    assert not df.duplicated(["idrssd", "qdate"]).any(), "Duplicate bank-quarter"
    assert df.qdate.mod(1).eq(0).all(), "Noninteger quarter code"
    assert df.year.eq(1960 + df.qdate.floordiv(4)).all(), "Year/quarter mismatch"
    assert df.qdate.between(96, 231).all(), "Quarter outside 1984Q1--2017Q4"
    audit = {"raw_rows": len(df), "raw_banks": int(df.idrssd.nunique())}

    # 3. Copy Q1 YTD amounts; for Q2--Q4 subtract the preceding calendar quarter's
    # YTD amount within the bank, masking gaps before any charter restriction.
    adjacent = df.groupby("idrssd").qdate.diff().eq(1)
    for source in ["interest_income", "interest_expense", "net_income"]:
        df[f"{source}_q"] = df.groupby("idrssd")[source].diff().where(adjacent).mask(
            df.qdate.mod(4).eq(0), df[source])

    # 4. Drop noncommercial charter rows only after their YTD information has
    # been used; mask invalid asset denominators while retaining rows for audit.
    df = df.loc[df.charter_type.eq(200)].reset_index(drop=True).copy()
    audit.update(commercial_rows=len(df), commercial_banks=int(df.idrssd.nunique()),
                 other_charter_rows=audit["raw_rows"] - len(df))
    assets = df.ave_asset.where(np.isfinite(df.ave_asset) & df.ave_asset.gt(0))
    audit["invalid_average_assets"] = int(assets.isna().sum())
    audit["internal_quarter_gaps"] = int(df.groupby("idrssd").qdate.diff().gt(1).sum())

    # 5. Annualize as 400 * quarterly flow / average assets, preserving raw flows.
    # Mask negative interest flows and infinite rates; retain zero flows and ROA losses.
    for name, source in zip(RATE_NAMES, ["interest_income", "interest_expense", "net_income"]):
        flow = df[f"{source}_q"]
        audit[f"negative_{source}_quarters"] = int(flow.lt(0).sum())
        if source != "net_income":
            flow = flow.where(flow.ge(0))
        rate = (400 * flow / assets).replace([np.inf, -np.inf], np.nan)
        df[f"{name}_rate_raw"] = rate

        # 6. Cap interest rates at each quarter's 99.5th percentile and ROA at its
        # 0.5th/99.5th percentiles; the sensitivity setting leaves valid rates uncapped.
        groups = rate.groupby(df.qdate)
        lower = groups.transform(lambda x: x.quantile(.005)) if name == "roa" and winsorize else None
        upper = groups.transform(lambda x: x.quantile(.995)) if winsorize else None
        cleaned = rate.clip(lower=lower, upper=upper) if winsorize else rate
        df[f"{name}_rate"] = cleaned
        audit[f"{name}_rates_clipped"] = int((rate.notna() & cleaned.ne(rate)).sum())
        audit[f"{name}_rates_unavailable"] = int(rate.isna().sum())

    # 7. Difference the final cleaned levels and attach counts of restrictions,
    # invalid values and clipping so subsequent outputs can explain the sample.
    df = add_differences(df)
    df.attrs["audit"] = audit
    return df

def f_construction():
    """Build complete quarterly policy changes and their three lags."""
    # 1. Select March, June, September and December monthly rates, as the prompt
    # defines the quarterly policy level using the quarter's final month.
    fed = pd.read_csv(PROJECT_ROOT / "data/raw/FedFunds.csv", parse_dates=["date"])
    fed = fed.loc[fed.date.dt.month.isin([3, 6, 9, 12])].copy()
    fed["quarter"] = fed.date.dt.to_period("Q")

    # 2. Include 1983 history and require one policy row per calendar quarter.
    # Stop on duplicate keys, unmatched/missing rates or a failed [0,100) plausibility check.
    f = pd.DataFrame({"quarter": pd.period_range("1983Q1", "2017Q4", freq="Q")})
    f = f.merge(fed[["quarter", "FedFunds"]], on="quarter", how="left",
                validate="one_to_one", indicator=True)
    assert f._merge.eq("both").all() and f.FedFunds.notna().all(), "Missing policy rate"
    assert f.FedFunds.between(0, 100, inclusive="left").all(), "Rate not in percent units"

    # 3. Difference on the complete quarterly calendar, then shift by one, two
    # and three quarters so the earliest 1984 regressions retain their policy history.
    f["FedFunds_dif"] = f.FedFunds.diff()
    for lag in (1, 2, 3):
        f[f"FedFunds_dif_lag{lag}"] = f.FedFunds_dif.shift(lag)

    # 4. Convert periods to the bank data's numeric quarter key and discard the
    # pre-1984 construction rows only after the changes and lags are available.
    f["qdate"] = (f.quarter.dt.year - 1960) * 4 + f.quarter.dt.quarter - 1
    return f.loc[f.qdate.ge(96)].drop(columns=["quarter", "_merge"])

def panel_construction(winsorize=True):
    """Merge the bank outcomes with quarter-level policy regressors."""
    # 1. Build the chosen outcome treatment and left-merge policy by quarter;
    # many-to-one validation permits many banks but only one policy record per date.
    banks = y_construction(winsorize)
    panel = banks.merge(f_construction(), on="qdate", how="left", validate="many_to_one", indicator=True)

    # 2. Stop if any bank quarter lacks a policy match, then remove the merge
    # indicator and preserve the construction audit alongside the complete panel.
    assert panel._merge.eq("both").all(), "Bank quarter has no policy observation"
    panel = panel.drop(columns="_merge")
    panel.attrs["audit"] = banks.attrs["audit"]
    return panel

def main():
    """Save the standalone panel-construction output."""
    # 1. Build the baseline panel and write a compressed interim CSV, creating
    # its directory if needed and fixing gzip's timestamp for reproducibility.
    panel = panel_construction()
    output = PROJECT_ROOT / "data/interim/part2_panel.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Saved {len(panel):,} bank-quarters to {output}")
    return panel

if __name__ == "__main__":
    main()

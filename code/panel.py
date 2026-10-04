"""Build Question 1's quarterly commercial-bank panel from the supplied files."""
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE_NAMES = ["interest_income", "interest_expense", "roa"]


def add_differences(df):
    """Differences require the immediately preceding calendar quarter."""
    adjacent = df.groupby("idrssd")["qdate"].diff().eq(1)
    for name in RATE_NAMES:
        df[f"{name}_dif"] = df.groupby("idrssd")[f"{name}_rate"].diff().where(adjacent)
    return df


def y_construction(winsorize=True):
    columns = ["idrssd", "year", "qdate", "charter_type", "bhc_id", "total_asset",
               "total_equity", "ave_asset", "interest_income", "interest_expense", "net_income"]
    df = pd.read_stata(PROJECT_ROOT / "data/raw/callreports_1984_2017.dta",
                      columns=columns, convert_dates=False).sort_values(["idrssd", "qdate"])
    assert df[["idrssd", "qdate"]].notna().all().all(), "Missing bank/date keys"
    assert df.idrssd.gt(0).all() and df.idrssd.mod(1).eq(0).all(), "Invalid bank ID"
    assert not df.duplicated(["idrssd", "qdate"]).any(), "Duplicate bank-quarter"
    assert df.qdate.mod(1).eq(0).all(), "Noninteger quarter code"
    assert df.year.eq(1960 + df.qdate.floordiv(4)).all(), "Year/quarter mismatch"
    assert df.qdate.between(96, 231).all(), "Quarter outside 1984Q1--2017Q4"
    audit = {"raw_rows": len(df), "raw_banks": int(df.idrssd.nunique())}
    # Q1 is already quarterly; Q2--Q4 need the preceding YTD report.
    adjacent = df.groupby("idrssd").qdate.diff().eq(1)
    for source in ["interest_income", "interest_expense", "net_income"]:
        df[f"{source}_q"] = df.groupby("idrssd")[source].diff().where(adjacent).mask(
            df.qdate.mod(4).eq(0), df[source])
    # Deaccumulate before filtering, so a preceding report is not discarded prematurely.
    df = df.loc[df.charter_type.eq(200)].reset_index(drop=True).copy()
    audit.update(commercial_rows=len(df), commercial_banks=int(df.idrssd.nunique()),
                 other_charter_rows=audit["raw_rows"] - len(df))
    assets = df.ave_asset.where(np.isfinite(df.ave_asset) & df.ave_asset.gt(0))
    audit["invalid_average_assets"] = int(assets.isna().sum())
    audit["internal_quarter_gaps"] = int(df.groupby("idrssd").qdate.diff().gt(1).sum())
    for name, source in zip(RATE_NAMES, ["interest_income", "interest_expense", "net_income"]):
        flow = df[f"{source}_q"]
        audit[f"negative_{source}_quarters"] = int(flow.lt(0).sum())
        if source != "net_income":
            flow = flow.where(flow.ge(0))  # YTD resets can produce negative interest; zero is valid.
        rate = (400 * flow / assets).replace([np.inf, -np.inf], np.nan)
        df[f"{name}_rate_raw"] = rate
        groups = rate.groupby(df.qdate)
        lower = groups.transform(lambda x: x.quantile(.005)) if name == "roa" and winsorize else None
        upper = groups.transform(lambda x: x.quantile(.995)) if winsorize else None
        cleaned = rate.clip(lower=lower, upper=upper) if winsorize else rate
        df[f"{name}_rate"] = cleaned
        audit[f"{name}_rates_clipped"] = int((rate.notna() & cleaned.ne(rate)).sum())
        audit[f"{name}_rates_unavailable"] = int(rate.isna().sum())
    df = add_differences(df)
    df.attrs["audit"] = audit
    return df


def f_construction():
    fed = pd.read_csv(PROJECT_ROOT / "data/raw/FedFunds.csv", parse_dates=["date"])
    fed = fed.loc[fed.date.dt.month.isin([3, 6, 9, 12])].copy()
    fed["quarter"] = fed.date.dt.to_period("Q")
    # Pre-1984 rates construct the initial policy change and all three lags.
    f = pd.DataFrame({"quarter": pd.period_range("1983Q1", "2017Q4", freq="Q")})
    f = f.merge(fed[["quarter", "FedFunds"]], on="quarter", how="left",
                validate="one_to_one", indicator=True)
    assert f._merge.eq("both").all() and f.FedFunds.notna().all(), "Missing policy rate"
    assert f.FedFunds.between(0, 100, inclusive="left").all(), "Rate not in percent units"
    f["FedFunds_dif"] = f.FedFunds.diff()
    for lag in (1, 2, 3):
        f[f"FedFunds_dif_lag{lag}"] = f.FedFunds_dif.shift(lag)
    f["qdate"] = (f.quarter.dt.year - 1960) * 4 + f.quarter.dt.quarter - 1
    return f.loc[f.qdate.ge(96)].drop(columns=["quarter", "_merge"])


def panel_construction(winsorize=True):
    banks = y_construction(winsorize)
    panel = banks.merge(f_construction(), on="qdate", how="left", validate="many_to_one", indicator=True)
    assert panel._merge.eq("both").all(), "Bank quarter has no policy observation"
    panel = panel.drop(columns="_merge")
    panel.attrs["audit"] = banks.attrs["audit"]
    return panel


def main():
    panel = panel_construction()
    output = PROJECT_ROOT / "data/interim/part2_panel.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Saved {len(panel):,} bank-quarters to {output}")
    return panel


if __name__ == "__main__":
    main()

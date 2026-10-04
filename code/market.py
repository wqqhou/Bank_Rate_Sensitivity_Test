"""Part 3C: dated bank/company links, a bank price index, and monthly returns."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DAYS = 120
MONEY = ["total_asset", "total_equity", "ave_asset", "net_income_q"]


def link_accounts(bank_panel, links):
    """Prefer a direct RSSD match; exclude competing companies at that priority."""
    columns = ["idrssd", "qdate", "bhc_id"] + MONEY
    bank = bank_panel[columns].sort_values(["idrssd", "qdate"]).reset_index(drop=True).copy()
    if bank.duplicated(["idrssd", "qdate"]).any():
        raise ValueError("Bank-quarter records must be unique before company aggregation.")
    bank["row"] = np.arange(len(bank))
    bank["quarter_end"] = pd.PeriodIndex.from_ordinals(
        bank.qdate.astype(int) - 40, freq="Q"
    ).to_timestamp(how="end").normalize()
    parent = bank.bhc_id.fillna(0)
    next_parent = parent.groupby(bank.idrssd).shift(-1)
    change = next_parent.notna() & next_parent.ne(parent)
    # Parent changes have no exact effective date: stop at the last old-parent report.
    bank["parent_until"] = bank.quarter_end.where(change).groupby(bank.idrssd).bfill()
    candidates = []
    for priority, identifier in enumerate(["idrssd", "bhc_id"]):
        part = bank.loc[bank[identifier].isin(links.rssd9001)].merge(
            links, left_on=identifier, right_on="rssd9001", how="inner"
        )
        part = part.loc[part.quarter_end.ge(part.dt_start) & part.quarter_end.le(part.dt_end)].copy()
        part["priority"] = priority
        candidates.append(part)
    matched = pd.concat(candidates, ignore_index=True)
    both = matched.groupby("row").priority.nunique().eq(2)
    matched = matched.loc[matched.priority.eq(matched.groupby("row").priority.transform("min"))].copy()
    ambiguous = matched.groupby("row").permco.nunique().gt(1)
    matched = matched.loc[~matched.row.isin(ambiguous.index[ambiguous])].copy()
    # Overlapping intervals for the SAME company are a continuous union at this quarter.
    selected = matched.groupby(["row", "permco"], as_index=False).agg(
        link_end=("dt_end", "max"), priority=("priority", "min"), rssd9001=("rssd9001", "first")
    )
    matched = bank.merge(selected, on="row", how="inner", validate="one_to_one")
    matched["valid_until"] = matched.link_end
    bhc = matched.priority.eq(1) & matched.parent_until.notna()
    matched.loc[bhc, "valid_until"] = matched.loc[bhc, ["link_end", "parent_until"]].min(axis=1)
    # Do not carry a record into a later conflicting company-link interval.
    changing_ids = links.groupby("rssd9001").permco.nunique()
    for rssd in changing_ids.index[changing_ids.gt(1)]:
        for link in links.loc[links.rssd9001.eq(rssd)].itertuples():
            mask = matched.rssd9001.eq(rssd) & matched.permco.ne(link.permco) & matched.quarter_end.lt(link.dt_start)
            matched.loc[mask, "valid_until"] = matched.loc[mask, "valid_until"].clip(upper=link.dt_start - pd.Timedelta(days=1))
    audit = {
        "bank_quarters": len(bank), "bank_quarters_matched": len(matched),
        "matched_banks_all_years": int(matched.idrssd.nunique()),
        "matched_companies_all_years": int(matched.permco.nunique()),
        "ambiguous_bank_quarters_excluded": int(ambiguous.sum()),
        "bank_quarters_with_both_link_routes": int(both.sum()),
        "direct_bank_quarters": int(matched.priority.eq(0).sum()),
        "holding_company_bank_quarters": int(matched.priority.eq(1).sum()),
    }
    return matched, audit

def company_accounts(matched):
    """Sum subsidiary amounts first; never average their accounting ratios."""
    m = matched.copy()
    m["valid_financials"] = np.isfinite(m[MONEY]).all(axis=1) & m.total_asset.gt(0) & m.ave_asset.gt(0)
    keys = ["permco", "qdate"]
    amounts = m.groupby(keys)[MONEY].sum(min_count=1)
    info = m.groupby(keys).agg(
        quarter_end=("quarter_end", "first"), valid_until=("valid_until", "min"),
        n_banks=("idrssd", "nunique"), complete=("valid_financials", "all")
    )
    company = amounts.join(info).reset_index()
    n_incomplete = int((~company.complete).sum())
    company = company.loc[company.complete].copy()
    company["roa_raw"] = 400 * company.net_income_q / company.ave_asset
    company["capital_raw"] = 100 * company.total_equity / company.total_asset
    company["log_assets"] = np.log(company.total_asset)
    for variable in ["roa", "capital"]:
        grouped = company.groupby("qdate")[f"{variable}_raw"]
        company[variable] = company[f"{variable}_raw"].clip(
            grouped.transform(lambda x: x.quantile(.005)),
            grouped.transform(lambda x: x.quantile(.995)),
        )
    company["available_date"] = company.quarter_end + pd.Timedelta(days=PUBLICATION_DAYS)
    return company, n_incomplete


def read_stock(companies):
    """Read selected fields in chunks and remove only verified duplicate records."""
    fields = ["PERMNO", "PERMCO", "YYYYMM", "MthCalDt", "MthRet", "MthRetx",
              "MthPrevCap", "MthPrevDt", "SecInfoStartDt", "SecInfoEndDt",
              "USIncFlg", "IssuerType", "SecurityType", "SecuritySubType", "ShareType", "PrimaryExch"]
    parts = []
    raw_rows = selected_rows = 0
    first_month, last_month = 999999, 0
    for chunk in pd.read_csv(ROOT / "data/raw/CRSP_Monthly_Stock_File.csv", usecols=fields, chunksize=200000):
        raw_rows += len(chunk)
        first_month = min(first_month, int(chunk.YYYYMM.min()))
        last_month = max(last_month, int(chunk.YYYYMM.max()))
        s = chunk.loc[chunk.PERMCO.isin(companies) & chunk.YYYYMM.between(198401, 201712)].copy()
        selected_rows += len(s)
        s["month"] = pd.to_datetime(s.YYYYMM.astype(str), format="%Y%m")
        for column in ["SecInfoStartDt", "SecInfoEndDt", "MthPrevDt"]:
            s[column] = pd.to_datetime(s[column])
        # Classify at the start of the holding month, retaining delisting-month returns.
        s = s.loc[s.month.ge(s.SecInfoStartDt) & s.month.le(s.SecInfoEndDt)]
        ordinary = s.USIncFlg.eq("Y") & s.IssuerType.isin(["CORP", "ACOR"])
        ordinary &= s.SecurityType.eq("EQTY") & s.SecuritySubType.eq("COM") & s.ShareType.eq("NS")
        ordinary &= s.PrimaryExch.isin(["N", "A", "Q"])
        parts.append(s.loc[ordinary])
    stock = pd.concat(parts, ignore_index=True)
    keys = ["PERMNO", "month"]
    core = ["PERMCO", "MthCalDt", "MthRet", "MthRetx", "MthPrevCap", "MthPrevDt"]
    duplicates = stock.loc[stock.duplicated(keys, keep=False)]
    if duplicates.groupby(keys)[core].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting monthly stock records require investigation.")
    n_duplicates = int(stock.duplicated(keys).sum())
    stock = stock.drop_duplicates(keys).copy()
    for column in ["MthRet", "MthRetx", "MthPrevCap"]:
        stock[column] = pd.to_numeric(stock[column], errors="coerce")
    valid = np.isfinite(stock[["MthRet", "MthRetx", "MthPrevCap"]]).all(axis=1)
    valid &= stock.MthPrevCap.gt(0) & stock.MthRet.ge(-1) & stock.MthRetx.ge(-1)
    valid &= stock.MthPrevDt.dt.to_period("M").eq(stock.month.dt.to_period("M") - 1)
    dropped = int((~valid).sum())
    stock = stock.loc[valid].copy()
    stock["weighted_return"] = stock.MthRet * stock.MthPrevCap
    stock["weighted_price_return"] = stock.MthRetx * stock.MthPrevCap
    # PERMCO is the company; multiple PERMNO share classes get lag-cap weights.
    company = stock.groupby(["PERMCO", "month"]).agg(
        lag_cap=("MthPrevCap", "sum"), return_numerator=("weighted_return", "sum"),
        price_numerator=("weighted_price_return", "sum"), n_share_classes=("PERMNO", "nunique")
    ).reset_index().rename(columns={"PERMCO": "permco"})
    company["return"] = company.return_numerator / company.lag_cap
    company["price_return"] = company.price_numerator / company.lag_cap
    audit = {"raw_stock_rows": raw_rows, "raw_stock_first_month": first_month,
             "raw_stock_last_month": last_month, "linked_stock_rows_before_filters": selected_rows,
             "duplicate_security_month_rows_removed": n_duplicates,
             "invalid_stock_security_months_removed": dropped, "usable_security_months": len(stock)}
    return company, audit

def available_panel(company, returns):
    """Use information available before the return begins; enforce link expiry."""
    joined = pd.merge_asof(
        returns.sort_values("month"), company.sort_values("available_date"),
        left_on="month", right_on="available_date", by="permco", direction="backward",
        allow_exact_matches=False,
    )
    month_end = joined.month + pd.offsets.MonthEnd(0)
    usable = joined.available_date.notna() & month_end.le(joined.valid_until)
    usable &= (joined.month - joined.quarter_end).dt.days.le(365)
    joined = joined.loc[usable].copy()
    if not joined.available_date.lt(joined.month).all():
        raise ValueError("Accounting information is not available before the return month.")
    return joined.sort_values(["permco", "month"])


def fit_return_regression(panel):
    from linearmodels.panel import PanelOLS
    variables = ["log_assets", "roa", "capital"]
    data = panel.set_index(["permco", "month"]).sort_index()
    fit = PanelOLS(100 * data["return"], data[variables], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True, low_memory=True
    )
    table = pd.DataFrame({"coefficient": fit.params, "std_error": fit.std_errors,
                          "t_stat": fit.tstats, "pvalue": fit.pvalues})
    table.index.name = "variable"
    # Adjusted R2 after accounting for the absorbed effects and the three slopes.
    adjusted = 1 - (1 - fit.rsquared_inclusive) * (fit.nobs - 1) / fit.df_resid
    return table, {"regression_nobs": int(fit.nobs), "regression_companies": int(panel.permco.nunique()),
                   "regression_months": int(panel.month.nunique()), "regression_r2": float(fit.rsquared_inclusive),
                   "regression_adjusted_r2": float(adjusted), "regression_within_r2": float(fit.rsquared_within)}


def run_market(bank_panel, output_dir, figure_dir):
    output_dir, figure_dir = Path(output_dir), Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(ROOT / "data/raw/Bank_and_CRSP_Linking_Table.csv", parse_dates=["dt_start", "dt_end"])
    links = links[["permco", "rssd9001", "dt_start", "dt_end"]].drop_duplicates()
    matched, audit = link_accounts(bank_panel, links)
    company, n_incomplete = company_accounts(matched)
    returns, stock_audit = read_stock(matched.permco.unique())
    panel = available_panel(company, returns)
    if panel.empty:
        raise ValueError("No company-months satisfy stock and accounting timing restrictions.")
    table, regression_info = fit_return_regression(panel)
    # A chain-linked price index uses ex-dividend returns, not raw price averages.
    index = panel.assign(numerator=panel.price_return * panel.lag_cap).groupby("month").agg(
        numerator=("numerator", "sum"), lag_cap=("lag_cap", "sum"), n_companies=("permco", "nunique")
    )
    index["price_return"] = index.numerator / index.lag_cap
    index["price_index"] = 100 * (1 + index.price_return).cumprod()
    index.index = index.index + pd.offsets.MonthEnd(0)
    baseline = pd.DataFrame({"price_index": [100.]}, index=[index.index.min() - pd.offsets.MonthEnd(1)])
    price_path = pd.concat([baseline, index[["price_index"]]]).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(price_path.index, price_path.price_index, color="#174d72", linewidth=1.7)
    ax.set(ylabel="Ex-dividend price index (base 100)", xlabel="", title="Publicly traded banks: market-capitalization-weighted prices")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    figure_path = figure_dir / "bank_stock_price_index.pdf"
    fig.savefig(figure_path)
    plt.close(fig)
    panel.to_csv(output_dir / "company_month_panel.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    company.to_csv(output_dir / "company_quarter_accounts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    index.to_csv(output_dir / "bank_stock_price_index.csv", index_label="month_end")
    table_path = output_dir / "stock_return_regression.csv"
    table.to_csv(table_path)
    used_pairs = panel[["permco", "qdate"]].drop_duplicates()
    used_banks = matched.merge(used_pairs, on=["permco", "qdate"], how="inner")
    result = {**audit, **stock_audit, **regression_info,
        "incomplete_company_quarters_excluded": n_incomplete,
        "matched_banks_in_stock_analysis": int(used_banks.idrssd.nunique()),
        "matched_companies_in_stock_analysis": int(panel.permco.nunique()),
        "first_return_month": str(panel.month.min().date()), "last_return_month": str(panel.month.max().date()),
        "index_last_value": float(index.price_index.iloc[-1]), "publication_delay_days": PUBLICATION_DAYS,
        "regression_table": str(table_path), "price_figure": str(figure_path),
        "method": "Direct RSSD links take priority over BHC links; ambiguous company links excluded. "
                  "Company accounts sum matched commercial-bank subsidiaries, not consolidated holding-company accounts. "
                  "Quarter-end +120 days is an assumed publication date, not an observed filing date. "
                  "Company and month fixed effects; SE clustered by company and month. "
                  "ROA and capital ratios winsorized at .5/99.5 percentiles within accounting quarter; returns untrimmed.",
        "crsp_reference": "https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/"}
    (output_dir / "market_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

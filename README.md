This project reproduces Question 1 of the bank interest-rate sensitivity exam. The final report is `report/main.pdf`; its source is `report/main.tex`.

Place the four supplied files in `data/raw/`:

- `callreports_1984_2017.dta`
- `FedFunds.csv`
- `Bank_and_CRSP_Linking_Table.csv`
- `CRSP_Monthly_Stock_File.csv`

Run these commands from the project root. Python 3.11 was used; the virtual-environment creation step is only needed if `code/venv` does not already exist. PDF compilation requires a LaTeX installation with `latexmk` and `pdflatex`.

```sh
python3.11 -m venv code/venv
code/venv/bin/python -m pip install -r code/requirements.txt
MPLCONFIGDIR=/tmp/fmc-matplotlib code/venv/bin/python code/run_question1.py
latexmk -pdf -cd report/main.tex
```

`run_question1.py` rebuilds the analysis from raw files, writes final datasets, audits and CSV/TeX tables to `data/question1/`, saves figures to `report/figures/`, and supplies numerical results through `report/numbers.tex`. Run the analysis before compiling the report. Files in `data/processed/` and `data/interim/` come from exploratory work; the final report uses `data/question1/`.

The main code files are:

- `panel.py`: validates bank-quarter keys, converts YTD reports to quarterly flows, constructs cleaned annualized rates and calendar-safe differences, and merges policy changes and lags.
- `estimation.py`: estimates each bank's income, expense and ROA betas and records sample eligibility.
- `bank_analysis.py`: produces the descriptive evidence and HC3 cross-sectional regressions, including size subsamples and binned scatter plots.
- `market.py`: applies dated bank/company links, aggregates accounting data and share classes, constructs the stock price index, and estimates the return regression.
- `run_question1.py`: runs all parts and the sensitivity check, and prepares the report inputs.

The baseline retains commercial-bank observations (`charter_type == 200`). Quarterly flows are constructed before that restriction. Rates require finite positive average assets; negative quarterly interest flows are unavailable, while zero interest and negative net income remain valid. Income and expense rates are capped at each quarter's 99.5th percentile; ROA is capped at its 0.5th and 99.5th percentiles. Differences are computed after this treatment. The sensitivity run omits these caps while retaining the other validity and sample rules.

Bank regressions require 40 usable quarters and include an intercept, the current policy change and three lags. Income and expense use a common sample; ROA uses its own usable observations. Beta is the sum of the four policy coefficients. Part 3A/B uses HC3 standard errors and splits banks at the analysis-specific median average assets.

The supplied stock file begins in 2000. Accounting information is assumed available 120 days after quarter-end and must precede the return month; actual filing dates are not supplied. The stock analysis combines bank reports at the listed-company level and accounts for dated links. Its methodology and limitations are described in the report.

`inference.py` and `summary.py` retain optional earlier inference utilities. Cumulative-response HAC tests are not needed to reproduce the main report. To generate them separately:

```sh
code/venv/bin/python code/estimation.py --inference --output-dir data/optional_inference
```

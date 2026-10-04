# Bank Interest-Rate Sensitivity — Submission

- `report/submission.pdf`: final report for Questions 1 and 2.
- `report/main.tex`, `report/question2.tex`: editable text; remaining TeX files and figures are generated.
- `code/`: analysis scripts, reproduction driver and pinned dependencies.
- `data/question1/`, `data/question2/`: estimates, tables, plotted series and sample audits. `question1/bank_betas.csv` is the required one-row-per-bank dataset.

Requires Python 3.11 and LaTeX with `latexmk`/`pdflatex`. Place the four supplied files in `data/raw/` (raw files are not bundled):

`callreports_1984_2017.dta`, `FedFunds.csv`, `Bank_and_CRSP_Linking_Table.csv`, `CRSP_Monthly_Stock_File.csv`.

From the submission folder, run:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python code/reproduce.py
```

This rebuilds both analyses from raw data and compiles the final PDF. Large intermediate panels are regenerated rather than bundled. Run `code/run_question1.py` or `code/question2.py` separately for either analysis. Filtering, inference and limitations are documented in the report.

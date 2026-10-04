"""Rebuild both analyses and compile the combined submission report."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_report():
    """Combine the editable question reports, keeping work times at the end."""
    # 1. Reuse the common LaTeX setup and each question's current text and time entry.
    report = ROOT / "report"
    sources = [(report / name).read_text() for name in ("main.tex", "question2.tex")]
    preamble = sources[0].split(r"\begin{document}", 1)[0]
    bodies, times = [], []
    for source in sources:
        body = source.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
        body, time = re.split(r"\\paragraph\{Time spent[^}]*\}", body, maxsplit=1)
        if "To be completed" in time or not time.strip():
            raise ValueError("Enter the required work time in each question's LaTeX source.")
        bodies.append(body.strip())
        times.append(time.strip())

    # 2. Keep one sequence of page, table and figure numbers in the final PDF.
    source = (preamble + "\\input{question2_numbers.tex}\n\\begin{document}\n"
              + "\n\\clearpage\n".join(bodies)
              + "\n\\par\\smallskip\n{\\small\\textbf{Time spent (reported).} Question 1: " + times[0]
              + "\\quad Question 2: " + times[1] + "}\n\\end{document}\n")
    (report / "submission.tex").write_text(source, encoding="utf-8")
    subprocess.run(["latexmk", "-pdf", "-silent", "-interaction=nonstopmode",
                    "-halt-on-error", "-cd", str(report / "submission.tex")], check=True)


def main():
    """Run every calculation from raw files before building the final report."""
    # 1. Use this interpreter and the project root so execution is location-independent.
    for script in ("run_question1.py", "question2.py"):
        subprocess.run([sys.executable, "-B", str(ROOT / "code" / script)], cwd=ROOT, check=True)

    # 2. Build the report only after both analyses finish successfully.
    build_report()
    print(f"Final report: {ROOT / 'report/submission.pdf'}")


if __name__ == "__main__":
    main()

"""Build the Assignment 01 PDF report.

Usage (from the repo root, venv active):
    pip install reportlab
    python scripts/build_report.py            # builds report.pdf
    python scripts/build_report.py --strict   # also exits with an error if anything is unfinished

What it does
    * draws the architecture diagram (screenshots/architecture.png)
    * pulls real numbers from your repo: dataset stats, MLflow runs, model metadata,
      number of tests, deliverables check
    * embeds screenshots from screenshots/ (see SHOT_CAPTIONS for the file names)
    * highlights in YELLOW everything still unfinished: your own text ([[...]]) and
      missing screenshots, and lists them in the terminal when it finishes.

Write your own words in the USER_TEXT block below. Then re-run the script as often as you like.
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
OUT_PDF = ROOT / "report.pdf"
EXPERIMENT_NAME = "heart-disease-classification"

# ==================================================================================
#  EDIT THIS BLOCK  -  your details and your own words
#  Anything still inside [[double square brackets]] is shown highlighted in yellow.
# ==================================================================================
STUDENT_NAME = "Dinesh Kumar Yadav"
STUDENT_ID = "2025CT05018"
GITHUB_URL = "https://github.com/Dinesh-Bits-2026/heart-disease-mlops"
VIDEO_URL = "https://youtu.be/D74Bc9mUGh4"
API_ACCESS = "http://localhost:8080 (local Kubernetes, Docker Desktop); see Section 8"

USER_TEXT = {
    "missing_values": (
        "The raw data had missing values only in ca (4 rows) and thal (2 rows), about 2% of the 303 "
        "records. They were filled with the column median (0 for ca and 3 for thal). In both columns "
        "the median is also the most frequent value, so no invalid category code is created. Filling "
        "instead of dropping keeps all 303 records, which matters in such a small dataset, and with "
        "only 6 values affected the effect on the results is negligible. The modelling pipeline "
        "additionally imputes any remaining gaps inside each cross-validation fold."
    ),
    "eda_takeaways": (
        "The target is fairly balanced (164 patients without disease and 139 with disease, 54.1% and "
        "45.9%), so accuracy is a fair metric. Precision, recall and ROC-AUC are reported as well "
        "because missing a sick patient costs more than a false alarm. oldpeak is strongly "
        "right-skewed with many zeros and chol is mildly right-skewed with a few very high values; "
        "no impossible values were found. The features most correlated with the target are thal "
        "(0.53), ca (0.46), exang (0.43), oldpeak (0.42) and cp (0.41) positively and thalach "
        "(-0.42) negatively, while fbs (0.03) and chol (0.09) carry almost no linear signal. The "
        "disease rates by category show the same picture: 22% of patients with thal = 3 have "
        "disease against 76% with thal = 7; 26% with no coloured vessels (ca = 0) against 85% with "
        "ca = 3; 73% with asymptomatic chest pain (cp = 4) against 18% to 30% for the other types; "
        "and 77% with exercise-induced angina against 31% without. Men (55%) have a higher disease "
        "rate than women (26%), whereas fasting blood sugar makes almost no difference (45% versus "
        "49%). Because the features are on very different scales and cp, thal, slope, restecg and "
        "ca are category codes, numeric features were standardised and the category codes were "
        "one-hot encoded."
    ),
    "model_choice": (
        "Logistic Regression was selected as the final model. In cross-validation on the training "
        "split it matched or slightly beat Random Forest on every metric (ROC-AUC 0.904 vs 0.900, "
        "recall 0.783 vs 0.774), so the two are effectively tied, and the simpler model is faster, "
        "easier to explain and less prone to overfitting on about 300 rows. The hold-out test set holds "
        "only about 61 patients, so one patient changes a score by roughly 1.6 points; the test scores "
        "(ROC-AUC 0.968 vs 0.949) are too noisy to overturn the cross-validation result. For a "
        "screening use case recall matters most: at the default 0.5 threshold roughly one in five "
        "sick patients is missed in cross-validation, and lowering the decision threshold would trade "
        "precision for recall."
    ),
    "tuning_notes": (
        "Both models were tuned with grid search on ROC-AUC, using 5-fold stratified cross-validation "
        "on the training split only: Logistic Regression over the regularisation strength C, and "
        "Random Forest over the number of trees, the maximum depth and the minimum samples per leaf. "
        "The hold-out set was kept untouched so that it gives an unbiased final check. For Logistic "
        "Regression the best and worst settings differed by 0.018 ROC-AUC (C = 1 gave 0.904 and "
        "C = 0.01 gave 0.886), so very strong regularisation hurt slightly while C from 0.1 to 10 "
        "gave almost the same result. For Random Forest all 12 settings fell within 0.008 ROC-AUC "
        "(0.892 to 0.900), so the model was insensitive to these hyper-parameters."
    ),
    "tracking_notes": (
        "Runs were compared in the MLflow UI by cross-validation ROC-AUC and recall, and the nested "
        "grid-search runs show how performance changed with each hyper-parameter setting. Training "
        "was run in four separate sessions between 4 and 18 October with the same fixed seed (42) "
        "and produced identical cross-validation and hold-out metrics every time (for example a "
        "cross-validation ROC-AUC of 0.904 for Logistic Regression and 0.900 for Random Forest), "
        "which confirms that the pipeline is reproducible. In the compare view the two models are "
        "within 0.004 ROC-AUC of each other in cross-validation; Random Forest has the higher "
        "hold-out accuracy (0.902 versus 0.885) but the lower hold-out ROC-AUC (0.949 versus 0.968), "
        "a mixed picture that is consistent with the small hold-out set."
    ),
    "reproducibility": (
        "To confirm that a clean setup works, I cloned the repository into a new folder, created a "
        "fresh Python 3.11 virtual environment, installed requirements.txt and requirements-dev.txt, "
        "and ran python -m pytest: all 29 tests passed. The GitHub Actions pipeline repeats the same "
        "steps from scratch on every push using Python 3.11 and passed, which is an independent "
        "check of the requirements file."
    ),
    "monitoring_notes": (
        "With normal traffic (200 requests from the traffic script) the dashboard showed 193 "
        "successful predictions and 7 rejected requests, matching the 200 sent; the 7 are "
        "deliberately invalid requests that the API rejected with HTTP 422. The p95 latency of "
        "/predict stayed at roughly 14 to 24 ms. The test traffic is randomly generated, so the "
        "outcome mix reflects the generator, not real patients. After the shifted batch the share "
        "of disease predictions went up from 75% to 92% and the average confidence rose from 0.78 to 0.87."
        "This is a simple illustration that a shift in incoming data shows up in prediction "
        "behaviour; it is not formal drift detection, which would compare feature distributions "
        "against the training data."
    ),
    "limitations": (
        "1. The dataset is small (about 300 patients from one study), so metrics are noisy and may "
        "not generalise; no real patient data was used and the model is a learning exercise, not a "
        "medical device. 2. The decision threshold is fixed at 0.5; a screening deployment would tune "
        "it for recall. 3. Kubernetes ran on a local single-node cluster using the local Docker image "
        "store, so there is no real high availability, autoscaling, cloud load balancer or container "
        "registry. 4. Monitoring covers service health and prediction behaviour only; there is no "
        "feature-level drift detection, alerting or automated retraining. Next steps would be to push "
        "the image to a container registry and deploy to a managed cluster (GKE, EKS or AKS), tune the "
        "decision threshold for recall, add feature-level drift detection with alerts, and automate "
        "retraining through the CI pipeline."
    ),
}


# Screenshot file names expected in the screenshots/ folder  ->  caption used in the report
SHOT_CAPTIONS = {
    "mlflow_runs.png": "MLflow experiment page listing the tracked runs.",
    "mlflow_compare.png": "MLflow comparison of runs (parameters and metrics side by side).",
    "mlflow_run_detail.png": "A single MLflow run: parameters, metrics and logged artifacts "
                             "(ROC curve, confusion matrix, model).",
    "ci_green.png": "GitHub Actions run with every step passing (lint, tests, training, "
                    "Docker build, container smoke test).",
    "ci_artifacts.png": "Artifacts produced by the workflow run.",
    "ci_red.png": "Deliberately failing run: the pipeline stops on a failing test and shows "
                  "a clear error log.",
    "docker_build.png": "Docker image build completing successfully.",
    "docker_run.png": "Container running locally with request log lines.",
    "api_predict.png": "Successful call to /predict returning prediction and confidence.",
    "api_docs.png": "Interactive API documentation (Swagger UI) at /docs.",
    "k8s_pods.png": "kubectl output: Deployment, 2 pods Running (1/1) and the LoadBalancer Service.",
    "k8s_service_test.png": "Health check and prediction through the Kubernetes service.",
    "prometheus_targets.png": "Prometheus targets page: API target is UP.",
    "grafana_dashboard.png": "Grafana dashboard with request, latency, prediction and "
                             "confidence panels.",
    "compose_logs.png": "API request logs (docker compose logs api).",
}
EDA_CAPTIONS = {
    "class_balance.png": "Class balance of the target.",
    "numeric_histograms.png": "Distributions of the numeric features.",
    "numeric_by_target.png": "Numeric features split by target.",
    "correlation_heatmap.png": "Correlation heatmap.",
    "categorical_vs_target.png": "Categorical features versus the target.",
}
# ==================================================================================
#  END OF EDIT BLOCK
# ==================================================================================

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

PLACEHOLDERS = []
MISSING_SHOTS = []
YELLOW = "#FFF59D"
NAVY = colors.HexColor("#1F3A5F")

base = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=base["Normal"], fontSize=10, leading=14, spaceAfter=6)
H1 = ParagraphStyle("H1x", parent=base["Heading1"], fontSize=15, textColor=NAVY,
                    spaceBefore=14, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("H2x", parent=base["Heading2"], fontSize=11.5, textColor=NAVY,
                    spaceBefore=8, spaceAfter=4, keepWithNext=1)
CAPTION = ParagraphStyle("Caption", parent=base["Normal"], fontSize=8.5, leading=11,
                         alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=8)
CELL = ParagraphStyle("Cell", parent=base["Normal"], fontSize=8.5, leading=11)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=7.8, leading=9.6,
                      backColor=colors.HexColor("#F3F3F3"), borderPadding=5, spaceAfter=8)
TOCTITLE = ParagraphStyle("TocTitle", parent=H1)
TITLE = ParagraphStyle("TitleX", parent=base["Title"], fontSize=26, leading=32, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=base["Normal"], fontSize=13, leading=18, alignment=TA_CENTER)


def hl(text: str) -> str:
    """Turn [[...]] into a yellow highlight and remember it as an open placeholder."""
    def sub(m):
        PLACEHOLDERS.append(m.group(1))
        return f'<font backColor="{YELLOW}">{m.group(1)}</font>'
    return re.sub(r"\[\[(.+?)\]\]", sub, text, flags=re.S)


def P(text, style=BODY):
    return Paragraph(hl(text), style)


def user(key):
    return P(escape(USER_TEXT[key]).replace("&amp;", "&amp;"))


def cell(text, style=CELL):
    return Paragraph(hl(text), style)


def code_block(text):
    return KeepTogether([Preformatted(text.strip("\n"), CODE)])


def code_section(title, text):
    return KeepTogether([Paragraph(title, H2), Preformatted(text.strip("\n"), CODE)])


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else cell(c, CELLB if (header and i == 0) else CELL)
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B7C3")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE4F0")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7F9FC")))
    t.setStyle(TableStyle(style))
    return t


SHOT_WIDTH, SHOT_MAX_H = 14.5 * cm, 7.5 * cm   # make smaller to shorten the report


def figure(path: Path, caption: str, width=SHOT_WIDTH, max_h=SHOT_MAX_H, label=None):
    label = label or path.name
    if path.exists():
        iw, ih = ImageReader(str(path)).getSize()
        scale = min(width / iw, max_h / ih, 1.0 if iw < 700 else 99)
        img = Image(str(path), iw * scale, ih * scale)
    else:
        MISSING_SHOTS.append(label)
        box = Table([[Paragraph(f'<b>Screenshot missing:</b> screenshots/{escape(label)}', CELL)]],
                    colWidths=[width], rowHeights=[1.8 * cm])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(YELLOW)),
                                 ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B59F00")),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        img = box
    return KeepTogether([img, Spacer(1, 3), Paragraph(escape(caption), CAPTION)])


def shot(name):
    return figure(SHOTS / name, SHOT_CAPTIONS[name])


def shot_row(names, folder=None, captions=None, width=7.9 * cm, max_h=5.4 * cm):
    """Up to two screenshots side by side."""
    folder = folder or SHOTS
    captions = captions or SHOT_CAPTIONS
    cells = []
    for n in names:
        path = folder / n
        label = n if folder == SHOTS else f"{folder.name}/{n}"
        if path.exists():
            iw, ih = ImageReader(str(path)).getSize()
            scale = min(width / iw, max_h / ih)
            img = Image(str(path), iw * scale, ih * scale)
        else:
            MISSING_SHOTS.append(label)
            img = Paragraph(f'<font backColor="{YELLOW}"><b>Screenshot missing:</b> '
                            f'screenshots/{escape(label)}</font>', CELL)
        cells.append([img, Spacer(1, 2), Paragraph(escape(captions.get(n, n)), CAPTION)])
    t = Table([cells], colWidths=[8.4 * cm] * len(cells))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return KeepTogether([t])


def shots(names):
    return [shot_row(names[i:i + 2]) for i in range(0, len(names), 2)]


# ---------------------------------------------------------------------------------
#  Data pulled from the repository
# ---------------------------------------------------------------------------------
def data_stats() -> dict:
    out = {}
    try:
        import pandas as pd
        clean = pd.read_csv(ROOT / "data" / "heart_clean.csv")
        vc = clean["target"].value_counts()
        out.update(rows=len(clean), cols=clean.shape[1],
                   neg=int(vc.get(0, 0)), pos=int(vc.get(1, 0)))
        raw = ROOT / "data" / "heart_raw.csv"
        if raw.exists():
            r = pd.read_csv(raw)
            out["missing"] = {c: int(n) for c, n in r.isna().sum().items() if n > 0}
    except Exception as exc:
        print("  (could not read dataset stats:", exc, ")")
    return out


def load_mlflow() -> dict:
    info = {}
    db = ROOT / "mlflow.db"
    if not db.exists():
        return info
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
        df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
        parents = df[df["params.model_type"].notna()].sort_values("start_time", ascending=False)
        latest = parents.drop_duplicates("params.model_type")
        info["n_parents"] = len(parents)
        info["n_children"] = len(df) - len(parents)
        info["latest"] = latest
    except Exception as exc:
        print("  (could not read MLflow runs:", exc, ")")
    return info


def load_metadata() -> dict:
    p = ROOT / "models" / "model_metadata.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def count_tests() -> int:
    n = 0
    for f in (ROOT / "tests").glob("test_*.py"):
        n += len(re.findall(r"^def test_", f.read_text(encoding="utf-8"), flags=re.M))
    return n


def read_text(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8").rstrip() if p.exists() else f"# {rel} not found"


# ---------------------------------------------------------------------------------
#  Architecture diagram
# ---------------------------------------------------------------------------------
def draw_architecture(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")
    palette = {"ml": "#DCE9F9", "ci": "#FDE7CC", "srv": "#DDF1DD", "mon": "#E9DDF5"}
    edge = {"ml": "#3B6EA8", "ci": "#C27A1A", "srv": "#3F8F3F", "mon": "#7C4EA3"}
    W, H = 15.0, 10.5

    def box(x, y, text, kind):
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=palette[kind], ec=edge[kind], lw=1.4))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, linespacing=1.35)

    def arrow(p1, p2, style="-|>", ls="-", label=None, lx=0, ly=1.4):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle=style, lw=1.3, color="#333333", ls=ls, shrinkA=0, shrinkB=0))
        if label:
            ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, fontsize=7.5,
                    ha="center", color="#333333", style="italic")

    row1, row2, row3 = 47, 27, 8
    xs = [9, 28, 47, 66, 85]
    box(xs[0], row1, "Data\nUCI Heart Disease\n(Cleveland CSV)", "ml")
    box(xs[1], row1, "EDA + cleaning\nnotebook,\ndownload_data.py", "ml")
    box(xs[2], row1, "Training\nsklearn Pipeline\nLogReg / RandomForest\n5-fold CV + tuning", "ml")
    box(xs[3], row1, "MLflow\nexperiment tracking\n+ model registry", "ml")
    box(xs[4], row1, "Packaged model\njoblib + metadata\n(models/)", "ml")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + W / 2 + 0.6, row1), (b - W / 2 - 0.6, row1))

    box(xs[4], row2, "Docker image\nFastAPI + model\nlogging, non-root", "srv")
    box(xs[3], row2, "Kubernetes\nDeployment (2 pods)\n+ LoadBalancer :8080", "srv")
    box(xs[2], row2, "Client\ncurl / Swagger /docs\nHTTP: POST /predict", "srv")
    arrow((xs[4], row1 - H / 2 - 0.6), (xs[4], row2 + H / 2 + 0.6))
    arrow((xs[4] - W / 2 - 0.6, row2), (xs[3] + W / 2 + 0.6, row2))
    arrow((xs[3] - W / 2 - 0.6, row2), (xs[2] + W / 2 + 0.6, row2), style="<|-|>")

    box(xs[0], row2, "GitHub repository\ncode, tests,\nmanifests", "ci")
    box(xs[1], row2, "GitHub Actions CI\nflake8, pytest, train,\ndocker build + smoke test", "ci")
    arrow((xs[0] + W / 2 + 0.6, row2), (xs[1] - W / 2 - 0.6, row2))
    arrow((xs[1] + 3, row2 + H / 2 + 0.6), (xs[2] - 3, row1 - H / 2 - 0.6), ls="--", label="runs training", lx=-6)

    box(xs[4], row3, "Prometheus\nscrapes /metrics\nevery 5 s", "mon")
    box(xs[3], row3, "Grafana\ndashboard\n(queries Prometheus)", "mon")
    arrow((xs[4], row2 - H / 2 - 0.6), (xs[4], row3 + H / 2 + 0.6), label="/metrics", lx=4, ly=0)
    arrow((xs[4] - W / 2 - 0.6, row3), (xs[3] + W / 2 + 0.6, row3))

    for i, (k, name) in enumerate([("ml", "ML pipeline"), ("ci", "CI/CD"), ("srv", "Serving"), ("mon", "Monitoring")]):
        x0 = 2 + i * 14
        ax.add_patch(FancyBboxPatch((x0, 2.5), 2.2, 2.2, boxstyle="round,pad=0.1", fc=palette[k], ec=edge[k]))
        ax.text(x0 + 3.2, 3.6, name, fontsize=8, va="center")
    ax.set_title("System architecture: heart-disease risk prediction (MLOps pipeline)", fontsize=12, pad=6)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------------
#  Report content
# ---------------------------------------------------------------------------------
class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1x":
            text = flowable.getPlainText()
            key = f"sec{abs(hash(text))}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, 0, 0)
            self.notify("TOCEntry", (0, text, self.page, key))


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.HexColor("#666666"))
    canv.drawString(2 * cm, 1.2 * cm, "MLOps Assignment 01 - Heart Disease Risk Prediction")
    canv.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canv.restoreState()


def build_story(stats, runs, meta):
    s = []
    # ---------------- cover ----------------
    s += [Spacer(1, 1.2 * cm), Paragraph("Heart Disease Risk Prediction", TITLE),
          Paragraph("End-to-End MLOps Pipeline", SUB), Spacer(1, 0.6 * cm),
          Paragraph("Machine Learning Operations (MLOps) - AIMLCZG523 - Assignment 01", SUB),
          Spacer(1, 0.8 * cm),
          table([["Student", P(STUDENT_NAME, CELL)], ["ID", P(STUDENT_ID, CELL)],
                 ["Programme", "M.Tech (AI/ML), BITS Pilani WILP"],
                 ["Code repository", P(GITHUB_URL, CELL)], ["Pipeline video", P(VIDEO_URL, CELL)],
                 ["Date", date.today().strftime("%d %B %Y")]],
                [4 * cm, 11 * cm], header=False, zebra=False),
          Spacer(1, 0.8 * cm)]

    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("TOC1", fontSize=10.5, leading=16, leftIndent=8)]
    s += [Paragraph("Contents", TOCTITLE), toc, PageBreak()]

    # ---------------- 1 introduction ----------------
    s.append(Paragraph("1. Introduction and Objective", H1))
    ds = ""
    if stats.get("rows"):
        ds = (f" The cleaned dataset has {stats['rows']} patient records and {stats['cols']} columns "
              f"(13 features plus the target); {stats['pos']} records ({stats['pos'] / stats['rows']:.1%}) "
              f"are labelled as heart disease present and {stats['neg']} ({stats['neg'] / stats['rows']:.1%}) "
              f"as absent.")
    s.append(P("The goal of this project is to build a classifier that predicts the risk of heart disease from "
               "patient health data and to deliver it as a reproducible, tested, containerised and monitored "
               "API, following MLOps practice. The data is the Cleveland subset of the UCI Heart Disease "
               "dataset (age, sex, chest pain type, blood pressure, cholesterol, ECG results, maximum heart "
               "rate and related measurements). The original 0-4 disease grade is converted to a binary "
               "target (0 = no disease, 1 = disease)." + ds))
    s.append(P("The project covers the full lifecycle: data acquisition and EDA, feature engineering, model "
               "training and tuning, experiment tracking with MLflow, packaging, automated testing and CI/CD "
               "with GitHub Actions, containerisation with Docker, deployment on Kubernetes, and monitoring "
               "with Prometheus and Grafana."))

    # ---------------- 2 architecture ----------------
    s.append(Paragraph("2. Architecture", H1))
    s.append(figure(SHOTS / "architecture.png", "Architecture of the pipeline (generated by scripts/build_report.py).",
                    width=16.5 * cm, max_h=9 * cm))
    s.append(P("Data flows left to right through the ML pipeline into a packaged model. The model is baked into a "
               "Docker image with a FastAPI service, which is deployed on Kubernetes and exposed through a "
               "LoadBalancer service. GitHub Actions lints, tests, trains and builds the image on every push. "
               "The service exposes Prometheus metrics, which Grafana visualises."))

    # ---------------- 3 setup ----------------
    s.append(Paragraph("3. Setup and Installation", H1))
    s.append(P("Requirements: Python 3.11, Git, Docker Desktop (with Kubernetes enabled for Section 8)."))
    s.append(code_block("""
git clone <repository-url>
cd heart-disease-mlops
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python src/download_data.py          # download + clean the dataset
python -m src.train                  # train, track with MLflow, save the model
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m pytest -v                  # unit tests
flake8 src api tests                 # lint
docker build -t heart-disease-api:v1 .
docker run --rm -p 8000:8000 heart-disease-api:v1
kubectl apply -f k8s/                # Kubernetes deployment
docker compose up -d                 # API + Prometheus + Grafana
python scripts/generate_traffic.py --n 200
"""))
    s.append(P("The repository layout is: <b>src/</b> (data, features, training, prediction), <b>api/</b> "
               "(FastAPI service), <b>tests/</b> (pytest), <b>k8s/</b> (manifests), <b>monitoring/</b> "
               "(Prometheus and Grafana configuration), <b>.github/workflows/</b> (CI), <b>notebooks/</b> "
               "(EDA) and <b>screenshots/</b>."))

    # ---------------- 4 data and EDA ----------------
    s.append(Paragraph("4. Data Acquisition and EDA", H1))
    s.append(P("<b>Acquisition.</b> <i>src/download_data.py</i> downloads the raw data from the UCI repository, "
               "assigns the column names, treats '?' as missing and writes <i>data/heart_raw.csv</i> and the "
               "cleaned <i>data/heart_clean.csv</i>."))
    miss = stats.get("missing")
    if miss:
        s.append(P("The raw data contained missing values in: " +
                   ", ".join(f"<i>{escape(k)}</i> ({v})" for k, v in miss.items()) + "."))
    s.append(Paragraph("Missing-value handling", H2))
    s.append(user("missing_values"))
    s.append(Paragraph("EDA findings", H2))
    s.append(user("eda_takeaways"))
    eda_dir = SHOTS / "eda"
    shown = set()
    names = list(EDA_CAPTIONS)
    for i in range(0, len(names), 2):
        s.append(shot_row(names[i:i + 2], folder=eda_dir, captions=EDA_CAPTIONS))
    shown.update(names)
    for extra in sorted(eda_dir.glob("*.png")) if eda_dir.exists() else []:
        if extra.name not in shown:
            s.append(figure(extra, extra.stem.replace("_", " ").capitalize() + " (own analysis).",
                            width=13 * cm, max_h=6.5 * cm))

    # ---------------- 5 modelling ----------------
    s.append(Paragraph("5. Feature Engineering and Model Development", H1))
    s.append(P("All preprocessing lives in one scikit-learn <i>ColumnTransformer</i> that is part of the "
               "model pipeline, so cross-validation never leaks information and the exact same transformations "
               "are applied at serving time. Numeric features (age, resting blood pressure, cholesterol, "
               "maximum heart rate, ST depression) are median-imputed and standardised. Categorical features "
               "(chest pain type, resting ECG, slope, thalassemia, number of vessels) are imputed with the most "
               "frequent value and one-hot encoded, ignoring unseen categories. Binary features (sex, fasting "
               "blood sugar, exercise-induced angina) are passed through after imputation."))
    cfg = ""
    latest = runs.get("latest")
    if latest is not None and len(latest):
        r0 = latest.iloc[0]
        cfg = (f" Evaluation uses a stratified {1 - float(r0.get('params.test_size', 0.2)):.0%}/"
               f"{float(r0.get('params.test_size', 0.2)):.0%} train/test split, {r0.get('params.cv_folds', '5')}-fold "
               f"stratified cross-validation, and a fixed random seed ({r0.get('params.seed', '42')}).")
    s.append(P("Two classifiers were trained and tuned with grid search on ROC-AUC: Logistic Regression and "
               "Random Forest." + cfg))
    s.append(Paragraph("Results", H2))
    if latest is not None and len(latest):
        hdr = ["Model", "CV acc.", "CV prec.", "CV recall", "CV ROC-AUC", "Test acc.", "Test prec.",
               "Test recall", "Test ROC-AUC"]
        keys = ["cv_accuracy", "cv_precision", "cv_recall", "cv_roc_auc", "test_accuracy",
                "test_precision", "test_recall", "test_roc_auc"]
        rows = [hdr]
        skip = {"model_type", "seed", "cv_folds", "test_size", "n_features"}
        tuned = []
        for _, r in latest.iterrows():
            rows.append([str(r["params.model_type"])] +
                        [f"{r.get('metrics.' + k, float('nan')):.3f}" for k in keys])
            ps = [f"{c[7:]}={r[c]}" for c in latest.columns
                  if c.startswith("params.") and c[7:] not in skip and str(r[c]) not in ("None", "nan")]
            tuned.append(f"<b>{escape(str(r['params.model_type']))}</b>: " + ", ".join(escape(p) for p in ps))
        s.append(table(rows, [2.6 * cm] + [1.7 * cm] * 8))
        s.append(Spacer(1, 4))
        s.append(P("Best hyper-parameters (latest runs): " + "; ".join(tuned) + "."))
    else:
        s.append(P("[[Results table not available: run python -m src.train so mlflow.db exists, then rebuild.]]"))
    s.append(Paragraph("Model selection", H2))
    final = meta.get("model")
    if final:
        s.append(P(f"The packaged final model is <b>{escape(str(final))}</b>."))
    s.append(user("model_choice"))
    s.append(user("tuning_notes"))

    # ---------------- 6 tracking ----------------
    s.append(Paragraph("6. Experiment Tracking with MLflow", H1))
    cnt = ""
    if runs.get("n_parents"):
        cnt = (f" The tracking database holds {runs['n_parents']} training runs and "
               f"{runs['n_children']} nested grid-search runs in the experiment "
               f"<i>{EXPERIMENT_NAME}</i>.")
    s.append(P("Every training run logs its parameters (model type, seed, folds, tuned hyper-parameters), "
               "cross-validation and hold-out metrics, the ROC curve and confusion matrix as plots, and the "
               "fitted pipeline with an input example and signature. Each hyper-parameter combination of the "
               "grid search is a nested child run, and the final model is tagged and registered in the MLflow "
               "model registry." + cnt))
    s.append(user("tracking_notes"))
    s += shots(["mlflow_runs.png", "mlflow_compare.png", "mlflow_run_detail.png"])

    # ---------------- 7 packaging ----------------
    s.append(Paragraph("7. Model Packaging and Reproducibility", H1))
    pk = ("The final pipeline (preprocessing plus model) is saved with joblib as <i>models/final_model.joblib</i>, "
          "next to <i>models/model_metadata.json</i>, which records the model name, MLflow run id, "
          "hyper-parameters, feature list, metrics and library versions. The same model is also stored as an "
          "MLflow model. Dependencies are pinned in <i>requirements.txt</i> (full project) and "
          "<i>requirements-api.txt</i> (serving image only).")
    s.append(P(pk))
    if meta.get("versions"):
        v = meta["versions"]
        s.append(P("Versions recorded at training time: " +
                   ", ".join(f"{escape(k)} {escape(str(x))}" for k, x in v.items()) + "."))
    s.append(user("reproducibility"))

    # ---------------- 8 ci/cd ----------------
    s.append(Paragraph("8. CI/CD Pipeline and Automated Testing", H1))
    s.append(P(f"The test suite contains {count_tests()} pytest tests covering data cleaning, the preprocessing "
               "pipeline, model training helpers, the prediction function and the API (valid requests, "
               "missing fields, out-of-range values, wrong types and metrics). The GitHub Actions workflow "
               "(<i>.github/workflows/ci.yml</i>) runs on every push and pull request:"))
    s.append(table([["Step", "Purpose"],
                    ["Install dependencies", "Clean environment from the requirements files"],
                    ["flake8", "Linting of src, api and tests"],
                    ["pytest", "Unit tests; JUnit report uploaded as an artifact"],
                    ["Train model", "Runs the training script; model and MLflow data uploaded as artifacts"],
                    ["Docker build", "Builds the serving image"],
                    ["Container smoke test", "Starts the container, checks /health and /predict"]],
                   [4.2 * cm, 11.8 * cm]))
    s.append(Spacer(1, 6))
    s.append(P("The pipeline fails on lint, test or build errors and prints clear logs, demonstrated below with a "
               "deliberately failing test."))
    s += shots(["ci_green.png", "ci_artifacts.png", "ci_red.png"])

    # ---------------- 9 docker / api ----------------
    s.append(Paragraph("9. Model Containerisation and API", H1))
    s.append(P("The model is served by a FastAPI application. Inputs are validated with allowed ranges, so "
               "invalid requests are rejected with HTTP 422. The image is based on python:3.11-slim, contains "
               "only the serving dependencies, runs as a non-root user and has a health check. Request logs "
               "record method, path, status and latency, and the prediction outcome, but never the patient's "
               "health data."))
    s.append(table([["Endpoint", "Description"],
                    ["POST /predict", "JSON patient record in; prediction, label, confidence and risk probability out"],
                    ["GET /health", "Readiness/liveness; confirms the model is loaded"],
                    ["GET /metrics", "Prometheus metrics"],
                    ["GET /model-info", "Packaged model metadata"]], [4 * cm, 12 * cm]))
    s.append(Spacer(1, 6))
    s.append(code_section("Example request", read_text("api/sample_request.json")))
    s.append(P("<i>confidence</i> is the probability of the predicted class; <i>risk_probability</i> is the "
               "probability of heart disease."))
    s += shots(["docker_build.png", "docker_run.png", "api_predict.png", "api_docs.png"])

    # ---------------- 10 kubernetes ----------------
    s.append(Paragraph("10. Production Deployment on Kubernetes", H1))
    s.append(P("The image is deployed to a local Kubernetes cluster (Docker Desktop) with a Deployment of two "
               "replicas, readiness and liveness probes on /health, and resource requests and limits. A "
               f"Service of type LoadBalancer exposes it. Access: {escape(API_ACCESS)}."))
    s.append(P("The complete manifests are in the k8s/ folder of the repository: <i>deployment.yaml</i> "
               "(2 replicas, probes, resource limits, Prometheus scrape annotations) and "
               "<i>service.yaml</i> (LoadBalancer, port 8080 to container port 8000)."))
    s += shots(["k8s_pods.png", "k8s_service_test.png"])

    # ---------------- 11 monitoring ----------------
    s.append(Paragraph("11. Monitoring and Logging", H1))
    s.append(P("The API exposes Prometheus metrics, scraped every 5 seconds. Docker Compose runs the API, "
               "Prometheus and Grafana together; Grafana loads a provisioned dashboard. A single API "
               "container is monitored so that counters are consistent (with several pods behind one "
               "service, each scrape would hit a different pod)."))
    s.append(table([["Metric", "Meaning"],
                    ["http_requests_total", "Requests by method, path and status code"],
                    ["http_request_duration_seconds", "Latency histogram per path"],
                    ["predictions_total", "Predictions by outcome (disease / no_disease)"],
                    ["prediction_confidence", "Histogram of prediction confidence"]], [6 * cm, 10 * cm]))
    s.append(Spacer(1, 6))
    s.append(user("monitoring_notes"))
    s.append(shot("grafana_dashboard.png"))
    s += shots(["prometheus_targets.png", "compose_logs.png"])

    # ---------------- 12 deliverables ----------------
    s.append(Paragraph("12. Repository, Deliverables and Limitations", H1))
    s.append(P(f"Code repository: {GITHUB_URL}"))
    checks = [("Code (src/, api/)", ["src/*.py", "api/*.py"]), ("Dockerfile", ["Dockerfile"]),
              ("requirements.txt", ["requirements.txt"]), ("Cleaned dataset", ["data/heart_clean.csv"]),
              ("Download script", ["src/download_data.py"]), ("Notebooks", ["notebooks/*.ipynb"]),
              ("Unit tests", ["tests/test_*.py"]), ("CI workflow", [".github/workflows/*.yml"]),
              ("Kubernetes manifests", ["k8s/*.yaml"]), ("Monitoring configuration", ["monitoring/prometheus.yml"]),
              ("Screenshots folder", ["screenshots/*.png"])]
    rows = [["Deliverable", "Status"]]
    for name, pats in checks:
        ok = all(any(ROOT.glob(p)) for p in pats)
        rows.append([name, "Present" if ok else "[[MISSING]]"])
    s.append(table(rows, [8 * cm, 4 * cm]))
    s.append(Spacer(1, 6))
    s.append(Paragraph("Limitations and future work", H2))
    s.append(user("limitations"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit with an error if anything is unfinished")
    args = ap.parse_args()

    SHOTS.mkdir(exist_ok=True)
    print("Drawing architecture diagram ...")
    draw_architecture(SHOTS / "architecture.png")
    print("Reading repository data ...")
    stats, runs, meta = data_stats(), load_mlflow(), load_metadata()
    print("Building PDF ...")
    story = build_story(stats, runs, meta)
    doc = ReportDoc(str(OUT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                    topMargin=2 * cm, bottomMargin=2 * cm,
                    title="MLOps Assignment 01 - Heart Disease Risk Prediction", author="")
    doc.multiBuild(story, onFirstPage=lambda c, d: None, onLaterPages=footer)

    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(OUT_PDF)).pages)
    except Exception:
        pages = "?"
    print(f"\nWrote {OUT_PDF.name}  ({pages} pages)")
    if MISSING_SHOTS:
        print(f"\nMissing screenshots ({len(MISSING_SHOTS)}):")
        for m in dict.fromkeys(MISSING_SHOTS):
            print("   screenshots/" + m)
    if PLACEHOLDERS:
        print(f"\nStill to write ({len(PLACEHOLDERS)} highlighted placeholders):")
        for p in dict.fromkeys(PLACEHOLDERS):
            print("   -", p[:90] + ("..." if len(p) > 90 else ""))
    if not MISSING_SHOTS and not PLACEHOLDERS:
        print("\nNothing left to fill in. Review the PDF once, start to finish.")
    if args.strict and (MISSING_SHOTS or PLACEHOLDERS):
        sys.exit(1)


if __name__ == "__main__":
    main()

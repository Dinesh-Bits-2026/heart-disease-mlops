"""One-time patch: makes build_report.py shorter (screenshots in pairs, cover + contents on one page,
no manifest listings). Your own text and settings in build_report.py are kept.

    python scripts/compact_report.py
A backup is saved as scripts/build_report.py.bak
"""
import shutil
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "build_report.py"
text = TARGET.read_text(encoding="utf-8")
if "def shot_row(" in text:
    print("Already compacted - nothing to do.")
    sys.exit(0)

NEW_FUNCS = '''def shot_row(names, folder=None, captions=None, width=7.9 * cm, max_h=5.4 * cm):
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


'''

REPLACEMENTS = [
    # helper functions go in front of the data-loading section
    ("# ---------------------------------------------------------------------------------\n"
     "#  Data pulled from the repository",
     NEW_FUNCS + "# ---------------------------------------------------------------------------------\n"
     "#  Data pulled from the repository"),
    # cover page and contents on one page
    ("Spacer(1, 4.5 * cm), Paragraph(\"Heart Disease Risk Prediction\", TITLE)",
     "Spacer(1, 1.2 * cm), Paragraph(\"Heart Disease Risk Prediction\", TITLE)"),
    ("          Spacer(1, 2.5 * cm),\n          table([[\"Student\"",
     "          Spacer(1, 0.8 * cm),\n          table([[\"Student\""),
    ("[4 * cm, 11 * cm], header=False, zebra=False),\n          PageBreak()]",
     "[4 * cm, 11 * cm], header=False, zebra=False),\n          Spacer(1, 0.8 * cm)]"),
    # EDA figures two per row
    ("    for name, cap in EDA_CAPTIONS.items():\n"
     "        s.append(figure(eda_dir / name, cap, width=13 * cm, max_h=6.5 * cm, label=f\"eda/{name}\"))\n"
     "        shown.add(name)\n",
     "    names = list(EDA_CAPTIONS)\n"
     "    for i in range(0, len(names), 2):\n"
     "        s.append(shot_row(names[i:i + 2], folder=eda_dir, captions=EDA_CAPTIONS))\n"
     "    shown.update(names)\n"),
    # screenshot groups
    ("    for n in [\"mlflow_runs.png\", \"mlflow_compare.png\", \"mlflow_run_detail.png\"]:\n        s.append(shot(n))\n",
     "    s += shots([\"mlflow_runs.png\", \"mlflow_compare.png\", \"mlflow_run_detail.png\"])\n"),
    ("    for n in [\"ci_green.png\", \"ci_artifacts.png\", \"ci_red.png\"]:\n        s.append(shot(n))\n",
     "    s += shots([\"ci_green.png\", \"ci_artifacts.png\", \"ci_red.png\"])\n"),
    ("    for n in [\"docker_build.png\", \"docker_run.png\", \"api_predict.png\", \"api_docs.png\"]:\n        s.append(shot(n))\n",
     "    s += shots([\"docker_build.png\", \"docker_run.png\", \"api_predict.png\", \"api_docs.png\"])\n"),
    ("    for n in [\"k8s_pods.png\", \"k8s_service_test.png\"]:\n        s.append(shot(n))\n",
     "    s += shots([\"k8s_pods.png\", \"k8s_service_test.png\"])\n"),
    ("    for n in [\"prometheus_targets.png\", \"grafana_dashboard.png\", \"compose_logs.png\"]:\n        s.append(shot(n))\n",
     "    s.append(shot(\"grafana_dashboard.png\"))\n"
     "    s += shots([\"prometheus_targets.png\", \"compose_logs.png\"])\n"),
    # manifests: point to the repository instead of printing them
    ("    s.append(code_section(\"k8s/deployment.yaml\", read_text(\"k8s/deployment.yaml\")))\n"
     "    s.append(code_section(\"k8s/service.yaml\", read_text(\"k8s/service.yaml\")))\n",
     "    s.append(P(\"The complete manifests are in the k8s/ folder of the repository: <i>deployment.yaml</i> \"\n"
     "               \"(2 replicas, probes, resource limits, Prometheus scrape annotations) and \"\n"
     "               \"<i>service.yaml</i> (LoadBalancer, port 8080 to container port 8000).\"))\n"),
]

for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        print("Could not apply this change (the script was edited in that spot):")
        print(old[:120].replace("\n", "\\n"))
        print("Nothing was changed. Send me this message.")
        sys.exit(1)
    text = text.replace(old, new)

shutil.copy(TARGET, str(TARGET) + ".bak")
TARGET.write_text(text, encoding="utf-8")
print("Done. Backup saved as build_report.py.bak - now run: python scripts/build_report.py")

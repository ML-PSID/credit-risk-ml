from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template

from analysis_web import build_storypack
from ml_web import build_ml_pack


BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "notebook" / "default of credit card clients.xls"

app = Flask(__name__)


@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/dashboard")
def dashboard() -> str:
    pack = build_storypack(str(DATASET))
    return render_template(
        "dashboard.html",
        metrics=pack.metrics,
        variable_dictionary=pack.variable_dictionary,
        charts=pack.charts,
    )


@app.route("/ml")
def ml() -> str:
    return render_template("ml.html")


@app.route("/ml-pack")
def ml_pack() -> str:
    pack = build_ml_pack(str(DATASET))
    return render_template(
        "ml_content.html",
        metrics=pack.metrics,
        rl_metrics=pack.rl_metrics,
        charts=pack.charts,
        best_params=pack.best_params,
        rl_best_params=pack.rl_best_params,
    )


if __name__ == "__main__":
    app.run(debug=True)

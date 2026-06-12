import io
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    from .inference_engine import app_dir, bundle_dir, predict_dataframe, resolve_paths
except ImportError:
    from inference_engine import app_dir, bundle_dir, predict_dataframe, resolve_paths


def template_dir() -> Path:
    if getattr(sys, "frozen", False):
        return bundle_dir() / "pangu_dispatcher_core" / "templates"
    return Path(__file__).resolve().parent / "templates"


app = Flask(__name__, template_folder=str(template_dir()))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

DEFAULT_CSV = os.environ.get(
    "GYMPULSE_CSV",
    str(app_dir() / "field_dispatch_dataa.csv") if getattr(sys, "frozen", False) else "/data/field_dispatch_dataa.csv",
)


def run_predictions(file_storage=None):
    _, ckpt_path = resolve_paths(DEFAULT_CSV)

    if file_storage is None:
        df = pd.read_csv(DEFAULT_CSV, encoding="utf-8-sig")
        source = Path(DEFAULT_CSV).name
    else:
        if not file_storage.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")
        df = pd.read_csv(io.BytesIO(file_storage.read()), encoding="utf-8-sig")
        source = file_storage.filename

    return predict_dataframe(df, ckpt_path), source


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predictions", methods=["GET", "POST"])
def predictions():
    try:
        upload = request.files.get("file") if request.method == "POST" else None
        payload, source = run_predictions(upload)
        return jsonify({
            "source": source,
            "count": len(payload),
            "predictions": payload,
        })
    except Exception as exc:
        app.logger.exception("Prediction request failed")
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "CSV file is too large. Maximum size is 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

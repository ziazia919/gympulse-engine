import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net


# Keep console output readable on Windows terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def configure_cpu():
    """Force CPU execution on both current and older MindSpore releases."""
    if hasattr(ms, "set_device"):
        try:
            ms.set_device(device_target="CPU", device_id=0)
        except TypeError:
            ms.set_device(device_target="CPU")
        ms.set_context(mode=ms.PYNATIVE_MODE)
        return

    try:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU", device_id=0)
    except TypeError:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")


configure_cpu()


STATUS_MATRIX = {
    0: {
        "status": "REJECTED",
        "log_template": (
            "CRITICAL DISPATCH ANOMALY: Deployment safety thresholds violated for "
            "operational target sector {location}. Local conditions [{terrain} | {weather}] "
            "present unmitigated system vulnerabilities. Task reassigned to standby pooling protocols."
        ),
        "authority": "Risk_Mitigation_Desk",
    },
    1: {
        "status": "APPROVED",
        "log_template": (
            "NEURAL DISPATCH VERIFIED: Autonomous routing vector successfully authorized for "
            "field target {location}. Environmental matrices [{terrain} | {weather}] converge "
            "safely within acceptable infrastructural risk thresholds. Commencing immediate resource allocation."
        ),
        "authority": "NOC_Technical_Director",
    },
}


class DispatchPredictor(nn.Cell):
    def __init__(self):
        super(DispatchPredictor, self).__init__()
        self.fc1 = nn.Dense(9, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Dense(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Dense(16, 2)

    def construct(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Folder containing GymPulseEngine.exe when frozen; project root when running as source."""
    if is_frozen():
        return Path(sys.executable).resolve().parent

    here = Path(__file__).resolve()
    if here.parent.name == "pangu_dispatcher_core":
        return here.parent.parent
    return here.parent


def bundle_dir() -> Path:
    """PyInstaller internal bundle folder, or source script folder in normal Python."""
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parent


def first_existing(candidates, label: str) -> Path:
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate

    msg = [f"Missing {label}. Checked these locations:"]
    msg.extend(f"  - {Path(p)}" for p in candidates)
    raise FileNotFoundError("\n".join(msg))


def resolve_paths(csv_override: str | None = None):
    external_dir = app_dir()
    internal_dir = bundle_dir()

    csv_candidates = []
    if csv_override:
        csv_candidates.append(Path(csv_override))
    csv_candidates.extend([
        external_dir / "field_dispatch_dataa.csv",          # preferred editable CSV beside exe
        internal_dir / "field_dispatch_dataa.csv",          # fallback if bundled
        Path.cwd() / "field_dispatch_dataa.csv",            # fallback when run from terminal
    ])

    ckpt_candidates = [
        internal_dir / "pangu_dispatcher_core" / "mindspore_tabular_predictor.ckpt",
        internal_dir / "mindspore_tabular_predictor.ckpt",
        external_dir / "mindspore_tabular_predictor.ckpt",
        Path.cwd() / "mindspore_tabular_predictor.ckpt",
    ]

    return first_existing(csv_candidates, "CSV file"), first_existing(ckpt_candidates, "CKPT model file")


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    required_columns = [
        "engineer_id",
        "engineer_name",
        "Location",
        "Terrain_Complexity",
        "Weather_Security",
        "route_security_risk",
        "ticket_urgency",
        "road_visibility",
        "engineer_Base_office",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV schema mismatch. Missing columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )

    terrain_map = {"Low Complexity": 1.0, "Medium Complexity": 2.0, "High Complexity": 3.0}
    weather_map = {"Low Severity": 1.0, "Medium Severity": 2.0, "High Severity": 3.0}
    risk_map = {"Low Risk": 1.0, "Medium Risk": 2.0, "High Risk": 3.0}
    urgency_map = {"Not urgent": 1.0, "Medium": 2.0, "Most Urgent": 3.0}
    visibility_map = {"Clear Visibility": 1.0, "Moderate Visibility": 2.0, "Low Visibility": 3.0}

    x = np.zeros((len(df), 9), dtype=np.float32)

    x[:, 0] = df["Terrain_Complexity"].astype(str).str.strip().map(terrain_map).fillna(2.0).values / 3.0
    x[:, 1] = df["Weather_Security"].astype(str).str.strip().map(weather_map).fillna(1.0).values / 3.0
    x[:, 2] = df["route_security_risk"].astype(str).str.strip().map(risk_map).fillna(1.0).values / 3.0
    x[:, 3] = df["ticket_urgency"].astype(str).str.strip().map(urgency_map).fillna(2.0).values / 3.0
    x[:, 4] = df["road_visibility"].astype(str).str.strip().map(visibility_map).fillna(1.0).values / 3.0

    offices = df["engineer_Base_office"].astype(str).str.strip().str.lower().fillna("islamabad office").values
    x[:, 5] = np.where(offices == "islamabad office", 1.0, 0.0)
    x[:, 6] = np.where(offices == "lahore office", 1.0, 0.0)
    x[:, 7] = np.where(offices == "karachi office", 1.0, 0.0)
    x[:, 8] = 0.5

    return x


def predict_dataframe(df: pd.DataFrame, ckpt_path: Path):
    df.columns = df.columns.str.strip()

    net = DispatchPredictor()
    param_dict = load_checkpoint(str(ckpt_path))
    load_param_into_net(net, param_dict)
    net.set_train(False)

    x = build_feature_matrix(df)

    input_tensor = Tensor(x, ms.float32)
    logits = net(input_tensor)

    softmax = nn.Softmax(axis=1)
    probabilities = softmax(logits).asnumpy()
    predicted_classes = np.argmax(probabilities, axis=1)

    output_json_payload = []

    for idx, row in df.iterrows():
        pred_class = int(predicted_classes[idx])
        conf_score = float(probabilities[idx][pred_class]) * 100
        config = STATUS_MATRIX[pred_class]

        formatted_log = config["log_template"].format(
            location=str(row["Location"]).strip(),
            terrain=str(row["Terrain_Complexity"]).strip(),
            weather=str(row["Weather_Security"]).strip(),
        )

        output_json_payload.append({
            "engineer_id": str(row["engineer_id"]).strip(),
            "engineer_name": str(row["engineer_name"]).strip(),
            "base_office": str(row["engineer_Base_office"]).strip(),
            "location": str(row["Location"]).strip(),
            "status": config["status"],
            "confidence": f"{conf_score:.2f}%",
            "prediction_text": formatted_log,
            "signoff_authority": config["authority"],
        })

    return output_json_payload


def run_dynamic_inference_pipeline(csv_override: str | None = None):
    csv_path, ckpt_path = resolve_paths(csv_override)

    print(f"CSV : {csv_path}")
    print(f"CKPT: {ckpt_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return predict_dataframe(df, ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="GymPulse MindSpore AI Predictor")
    parser.add_argument("--csv", help="Optional path to field_dispatch_dataa.csv", default=None)
    parser.add_argument(
        "--output-dir",
        help="Directory for predictions_output.json and predictions_output.csv",
        default=None,
    )
    parser.add_argument("--pause", action="store_true", help="Keep terminal open after execution")
    args = parser.parse_args()

    try:
        payload = run_dynamic_inference_pipeline(args.csv)

        out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else app_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "predictions_output.json"
        csv_out_path = out_dir / "predictions_output.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        pd.DataFrame(payload).to_csv(csv_out_path, index=False, encoding="utf-8-sig")

        print(f"\nDynamic data tokenizer finished packaging {len(payload)} active engineer profiles.")
        print(f"JSON output: {json_path}")
        print(f"CSV output : {csv_out_path}")

        print("\nPreview:")
        for row in payload[:5]:
            print(f"- {row['engineer_id']} | {row['engineer_name']} | {row['location']} | {row['status']} | {row['confidence']}")

    except Exception as exc:
        print("\nERROR: GymPulseEngine failed.")
        print(str(exc))
        sys.exit(1)
    finally:
        if args.pause:
            input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()

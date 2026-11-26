import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
CATEGORICAL = ["age", "admission_type_id", "diabetesMed", "change", "insulin", "A1Cresult"]
DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}


def load_encounters(path):
    data = pd.read_csv(path, na_values="?", keep_default_na=False)
    data = data.drop_duplicates("encounter_id")
    eligible = ~data["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)
    known_target = data["readmitted"].isin(["<30", ">30", "NO"])
    return data.loc[eligible & known_target & data["patient_nbr"].notna()].reset_index(drop=True)


def prepare_features(data):
    result = data[NUMERIC + CATEGORICAL].copy()
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in CATEGORICAL:
        result[column] = result[column].astype("string").fillna("Missing").astype(str)
    return result


def build_model(estimator):
    encoder = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return make_pipeline(encoder, estimator)


def score_model(name, actual, scores):
    slots = max(1, int(0.10 * len(actual)))
    # A fixed shuffle breaks equal-score ties without consulting outcomes.
    order = np.random.default_rng(42).permutation(len(actual))
    ranked = order[np.argsort(-scores[order], kind="stable")]
    caught = int(actual[ranked[:slots]].sum())
    return (name, average_precision_score(actual, scores), roc_auc_score(actual, scores),
            brier_score_loss(actual, scores), slots, caught, caught / slots, caught / actual.sum())


def plot_calibration(actual, predictions, output):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    end = 0.25
    for name, scores in predictions.items():
        observed, predicted = calibration_curve(actual, scores, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, "o-", label=name)
        end = min(1, max(end, 1.05 * max(observed.max(), predicted.max())))
    ax.plot([0, end], [0, end], "--", color="gray", label="Perfect calibration")
    ax.set(xlim=(0, end), ylim=(0, end), xlabel="Mean predicted probability",
           ylabel="Observed readmission rate", title="Calibration on held-out patients")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare readmission models on unseen patients.")
    parser.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_encounters(args.data)
    y = data["readmitted"].eq("<30").to_numpy(dtype=int)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train, test = next(split.split(data, y, groups=data["patient_nbr"]))
    train_patients = set(data.iloc[train]["patient_nbr"])
    test_patients = set(data.iloc[test]["patient_nbr"])
    if train_patients & test_patients:
        raise RuntimeError("The same patient appeared in both splits.")
    features = prepare_features(data)
    estimators = {
        "Training prevalence": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=500),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.06, early_stopping=False, random_state=42,
        ),
    }
    scores, rows = {}, []
    with threadpool_limits(limits=4):
        for name, estimator in estimators.items():
            model = build_model(estimator)
            model.fit(features.iloc[train], y[train])
            scores[name] = model.predict_proba(features.iloc[test])[:, 1]
            rows.append(score_model(name, y[test], scores[name]))
            if name == "Gradient boosting":
                joblib.dump(model, args.output / "readmission_model.joblib")
    metrics = pd.DataFrame(rows, columns=[
        "model", "average_precision", "roc_auc", "brier_score", "review_slots",
        "readmissions_caught", "precision_at_capacity", "recall_at_capacity",
    ])
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame([
        ("train", len(train), len(train_patients), int(y[train].sum())),
        ("test", len(test), len(test_patients), int(y[test].sum())),
    ], columns=["split", "encounters", "patients", "readmitted_within_30_days"]).to_csv(
        args.output / "split_summary.csv", index=False,
    )
    plot_calibration(y[test], scores, args.output)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

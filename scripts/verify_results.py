"""Verify saved matched results using only the Python standard library.

No model fitting, API requests, or new experimental outcomes are produced.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "research_v2/evaluation/classification"
CLM = "outputs/study/clm/kaggle_import/evaluation/classification"
CLM_METHODS = ("clm_direct_choice", "clm_choice_probabilities", "clm_geometry_questions",
               "clm_all_questions", "geometry_plus_clm_choice_probabilities",
               "geometry_plus_clm_geometry_questions", "geometry_plus_clm", "descriptor_plus_clm")


def read_bytes(name: str) -> bytes:
    p = ROOT / name
    if p.is_file():
        return p.read_bytes()
    return gzip.decompress(Path(str(p) + ".gz").read_bytes())


def read_json(name: str):
    return json.loads(read_bytes(name))


def read_csv(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(read_bytes(name).decode("utf-8-sig"))))


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def classification(rows: list[dict], labels: list[str]) -> dict:
    n = len(rows)
    correct, truth, predicted, true_positive = 0, Counter(), Counter(), Counter()
    nll, brier = 0.0, 0.0
    for row in rows:
        y, pred = row["label"], row["prediction"]
        require(y in labels and pred in labels, "Unknown classification label")
        truth[y] += 1
        predicted[pred] += 1
        correct += y == pred
        if y == pred:
            true_positive[y] += 1
        p = [min(1.0, max(1e-12, float(row["prob__" + label]))) for label in labels]
        require(all(math.isfinite(v) for v in p), "Nonfinite probability")
        total = sum(p)
        p = [v / total for v in p]
        nll -= math.log(max(p[labels.index(y)], 2.220446049250313e-16))
        brier += sum((value - (label == y)) ** 2 for label, value in zip(labels, p))
    return {"n": n, "accuracy": correct / n,
            "macro_f1": sum(2 * true_positive[y] / (truth[y] + predicted[y])
                            if truth[y] + predicted[y] else 0 for y in labels) / len(labels),
            "balanced_accuracy": sum(true_positive[y] / truth[y] for y in labels) / len(labels),
            "nll": nll / n, "brier": brier / n}


def verify(write_outputs: bool = False) -> dict:
    manifest = read_json("ARTIFACT_PROVENANCE.json")
    for artifact in manifest["artifacts"]:
        data = (ROOT / artifact["path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == artifact["sha256"], "Artifact hash mismatch: " + artifact["path"])
    cohort = read_csv("outputs/study/full/cohort.csv")
    by_id = {r["sample_id"]: r for r in cohort}
    counts = dict(Counter(r["split"] for r in cohort))
    require(len(by_id) == len(cohort) == 9798, "Frozen cohort identity mismatch")
    require(counts == {"train": 6858, "val": 980, "test": 1960}, "Frozen split counts changed")
    test = {sid: row for sid, row in by_id.items() if row["split"] == "test"}
    labels = read_json(BASE + "/label_mapping.json")["class_order"]
    base_summary = read_json(BASE + "/summary.json")
    clm_summary = read_json(CLM + "/summary.json")
    require(base_summary["cohort_hash"] == clm_summary["cohort_hash"], "CLM/base cohort hashes differ")
    summaries = {**base_summary["methods"], **{m: clm_summary["methods"][m] for m in CLM_METHODS}}
    predictions, metrics = {}, {}
    for method, expected in summaries.items():
        directory = CLM if method in CLM_METHODS else BASE
        rows = read_csv(directory + "/" + method + ".csv")
        indexed = {r["sample_id"]: r for r in rows}
        require(len(indexed) == len(rows) == 1960 and set(indexed) == set(test), "Unpaired test IDs: " + method)
        require(all(r["split"] == "test" and r["label"] == test[sid]["label"] for sid, r in indexed.items()),
                "Test label/split mismatch: " + method)
        actual = classification(rows, labels)
        for key, value in actual.items():
            require(abs(value - expected[key]) < 2e-6, f"Metric mismatch: {method}/{key}: {value} vs {expected[key]}")
        predictions[method] = indexed
        metrics[method] = actual
    # CLM's repeated reference rows, when supplied, must agree with baseline labels/predictions.
    repeated = set(clm_summary["methods"]) & set(base_summary["methods"])
    repeated_differences = {}
    for method in sorted(repeated):
        rows = read_csv(CLM + "/" + method + ".csv")
        require(len(rows) == len(test) and {r["sample_id"] for r in rows} == set(test), "Imported reference IDs differ")
        require(all(r["label"] == predictions[method][r["sample_id"]]["label"] for r in rows), "Imported reference labels differ")
        differences = sum(r["prediction"] != predictions[method][r["sample_id"]]["prediction"] for r in rows)
        repeated_differences[method] = differences
        # The imported descriptor-only rerun is retained as historical output.
        # The manuscript and joined table use the original baseline, consistently.
        require(differences == (2 if method == "descriptor_only" else 0), "Unexpected imported reference change: " + method)
    audit = read_json("outputs/study/full/evaluation/duplicate_audit.json")
    compromised = {str(sample["sample_id"]) for group in audit["groups"]
                   if "test" in {s["split"] for s in group["samples"]}
                   and {s["split"] for s in group["samples"]} & {"train", "val"}
                   for sample in group["samples"] if sample["split"] == "test"}
    require(len(compromised) == 132, "Duplicate sensitivity membership changed")
    retrieval_dir = "outputs/study/visual_proxy_evaluation/retrieval"
    retrieval_summary = read_json(retrieval_dir + "/summary.json")
    require(retrieval_summary.get("visual_proxy") is True, "Missing retrieval provenance label")
    retrieval = {}
    query_ids = None
    for method, expected in retrieval_summary["methods"].items():
        rows = [r for r in read_csv(retrieval_dir + "/" + method + "_per_query.csv") if by_id[r["query_id"]]["split"] == "test"]
        ids = {r["query_id"] for r in rows}
        require(len(rows) == len(ids) == expected["n_queries"] == 100, "Retrieval query count changed")
        require(query_ids is None or ids == query_ids, "Retrieval methods use different queries")
        query_ids = ids
        actual = {key: sum(float(r[key]) for r in rows) / len(rows) for key in expected if key in rows[0]}
        for key in actual:
            require(abs(actual[key] - expected[key]) < 1e-10, "Retrieval aggregate mismatch: " + method + "/" + key)
        retrieval[method] = actual
    # The packaged forms are blank protocol templates, never supplied expert results.
    for role in ["rater_a", "rater_b", "adjudication"]:
        rows = read_csv("outputs/study/full/annotations/" + role + ".csv")
        require(all(not r["grade"].strip() for r in rows), "Unexpected grades in template " + role)
    report = {"status": "passed", "cohort_size": len(cohort), "split_counts": counts,
              "paired_classification_methods": len(predictions), "test_samples_per_method": len(test),
              "duplicate_overlap_test_n": len(compromised), "retrieval_methods": len(retrieval),
              "imported_reference_prediction_differences": repeated_differences,
              "retrieval_test_queries_per_method": 100,
              "retrieval_label_source": retrieval_summary["relevance"],
              "classification": metrics, "retrieval": retrieval,
              "scope": "Saved-output verification only; no new training, inference, or expert annotation"}
    if write_outputs:
        out = ROOT / "results"
        out.mkdir(exist_ok=True)
        (out / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (out / "paired_test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["sample_id", "split", "label", "relative_path", *["prediction__" + m for m in predictions]]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for sid in sorted(test):
                writer.writerow({**{k: test[sid][k] for k in fields[:4]},
                                 **{"prediction__" + m: predictions[m][sid]["prediction"] for m in predictions}})
        with (out / "classification_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "n", "accuracy", "macro_f1", "balanced_accuracy", "nll", "brier"], lineterminator="\n")
            writer.writeheader()
            writer.writerows({"method": m, **values} for m, values in metrics.items())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write joined predictions and recalculated reports under results/")
    args = parser.parse_args()
    report = verify(args.write)
    print(json.dumps({k: v for k, v in report.items() if k not in {"classification", "retrieval"}}, indent=2))

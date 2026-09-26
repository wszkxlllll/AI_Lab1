"""从逐次运行记录生成汇总表和报告数字。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_results(directory=ROOT / "results"):
    records = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    rows = []
    for record in records:
        row = {k: v for k, v in record.items()
               if k not in {"spec", "classification_report", "confusion_matrix", "loss_curve", "warnings"}}
        row.update({k: v for k, v in record["spec"].items() if k != "params"})
        row["params"] = json.dumps(record["spec"]["params"], sort_keys=True)
        row["warning"] = " | ".join(record["warnings"])
        rows.append(row)
    return records, pd.DataFrame(rows)


def best_rows(frame, groups):
    return (frame.sort_values(["macro_f1", "accuracy", "fit_seconds"], ascending=[False, False, True])
            .groupby(groups, sort=False).head(1))


def tables(directory=ROOT / "results"):
    records, frame = load_results(directory)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    seed = config["seed"]
    primary = frame[frame.seed.eq(seed)]
    comparison = primary[primary.stage.eq("comparison") & primary.representation.isin(["bow", "tfidf"])]
    main = best_rows(comparison, ["model", "representation"]).sort_values(["model", "representation"])
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    structure = primary[primary.stage.eq("structure") | primary.run_id.eq(selection["structure_anchor_run_id"])]
    capacity = primary[primary.model.eq("MLP") & primary.representation.eq("tfidf") & primary.view.eq("joined")]
    repeated = []
    for spec, group in frame.groupby(["model", "representation", "view", "title_weight", "params"]):
        if group.seed.nunique() < 2:
            continue
        repeated.append(dict(zip(["model", "representation", "view", "title_weight", "params"], spec)) | dict(
            n_splits=len(group), macro_f1_mean=group.macro_f1.mean(), macro_f1_sd=group.macro_f1.std(ddof=1),
            accuracy_mean=group.accuracy.mean(), accuracy_sd=group.accuracy.std(ddof=1),
        ))
    repeat = pd.DataFrame(repeated)
    for name, table in {"all_runs": frame, "model_comparison": main, "structure": structure,
                        "capacity": capacity, "sensitivity": repeat}.items():
        table.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8")
    chosen = next(r for r in records if r["run_id"] == selection["selected_run_id"])
    per_class = pd.DataFrame({k: v for k, v in chosen["classification_report"].items() if isinstance(v, dict)}).T
    per_class.to_csv(directory / "selected_classification_report.csv", encoding="utf-8")
    np.savetxt(directory / "selected_confusion_matrix.csv", chosen["confusion_matrix"], delimiter=",", fmt="%d")
    return dict(all_runs=frame, main=main, structure=structure, capacity=capacity, sensitivity=repeat)


def parameter_label(row):
    params = json.loads(row.params)
    if row.model == "MLP":
        hidden = "+".join(map(str, params["hidden"]))
        return f"{hidden}, $\\alpha={params['alpha']:g}$"
    key = "alpha" if row.model == "NB" else "C"
    symbol = "\\alpha" if key == "alpha" else "C"
    return f"${symbol}={params[key]:g}$"


def write_report_numbers(directory=ROOT / "results"):
    result = tables(directory)
    records, frame = load_results(directory)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    rows = []
    for representation in ("bow", "tfidf"):
        for model in ("NB", "SVM", "LR", "MLP"):
            rows.append(result["main"].query("representation == @representation and model == @model").iloc[0])
    w2v = frame[(frame.seed == config["seed"]) & (frame.representation == "word2vec")]
    rows.append(w2v.sort_values(["macro_f1", "accuracy", "fit_seconds"], ascending=[False, False, True]).iloc[0])
    labels = dict(bow="BoW", tfidf="TF-IDF", word2vec="Word2Vec")
    lines = []
    for row in rows:
        seconds = f"{row.fit_seconds:.2f}" if row.fit_seconds >= 0.01 else "$<0.01$"
        lines.append(f"{labels[row.representation]} & {row.model} & {parameter_label(row)} & "
                     f"{row.accuracy*100:.2f} & {row.macro_f1*100:.2f} & {seconds} " + r"\\")
    lines[-1] = lines[-1][:-2].rstrip()
    (ROOT / "report" / "main_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    chosen = next(r for r in records if r["run_id"] == selection["selected_run_id"])
    commands = {"SelectedFone": f"{chosen['macro_f1']*100:.2f}",
                "SelectedAccuracy": f"{chosen['accuracy']*100:.2f}", "NumberRuns": str(len(records))}
    (ROOT / "report" / "numbers.tex").write_text(
        "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in commands.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = tables()
    write_report_numbers()
    print(result["main"][["model", "representation", "params", "accuracy", "macro_f1", "fit_seconds"]].to_string(index=False))

"""按预先确定的顺序运行实验，并保存每组配置的结果。"""

import hashlib
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import load_training, stratified_indices, validation_mask
from io_utils import save_json, file_hash
from models import make_model, specification
from text_features import TextFeatures, split_message

ROOT = Path(__file__).resolve().parent

class Experiment:
    def __init__(self, data_dir, output, config):
        self.data_dir, self.output, self.config = data_dir, output, config
        output.mkdir(parents=True, exist_ok=True)
        train_path = data_dir / "train_data.csv"
        self.data = load_training(train_path)
        self.y = self.data.target.to_numpy(dtype=int)
        self.texts = self.data.text.to_numpy()
        self.fingerprint = dict(
            data_sha256=file_hash(train_path), config=config,
            code_sha256={p.name: file_hash(p) for p in [ROOT / name for name in ("run_experiments.py", "search.py", "models.py", "data.py", "io_utils.py", "text_features.py")]},
        )
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old["fingerprint"] != self.fingerprint:
                raise ValueError("Data, configuration or code changed; use a new output directory")
        else:
            save_json(manifest_path, dict(
                fingerprint=self.fingerprint, python=platform.python_version(), platform=platform.platform(),
                sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
                logical_cpus=os.cpu_count(),
            ))
        self.records = []
        self.cache = {}
        self.seen = set()
        parts = [split_message(t) for t in self.texts]
        save_json(output / "data_audit.json", dict(
            samples=len(self.y), class_counts=self.data.target.value_counts().sort_index().to_dict(),
            exact_duplicate_extra_rows=int(self.data.text.duplicated().sum()),
            empty_titles=sum(not p[0].strip() for p in parts),
            empty_bodies=sum(not p[1].strip() for p in parts),
        ))

    def split(self, seed):
        train, valid = stratified_indices(self.y, self.config["validation_fraction"], seed)
        included = validation_mask(self.texts, train, valid)
        save_json(self.output / "splits" / f"seed_{seed}.json", dict(
            train=train.tolist(), validation=valid.tolist(), excluded_validation=valid[~included].tolist()))
        return train, valid

    def run(self, spec, seed=66, stage="comparison"):
        key = hashlib.sha256(json.dumps(dict(spec=spec, seed=seed), sort_keys=True).encode()).hexdigest()[:16]
        record_path = self.output / "runs" / f"{key}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            train, valid = self.split(seed)
            included = validation_mask(self.texts, train, valid)
            feature_key = (spec["representation"], spec["view"], spec["title_weight"], seed)
            if feature_key not in self.cache:
                features = TextFeatures(self.config, spec["representation"], spec["view"], spec["title_weight"], seed)
                start = time.perf_counter()
                x_train = features.fit_transform(self.texts[train])
                x_valid = features.transform(self.texts[valid])
                feature_seconds = time.perf_counter() - start
                self.cache[feature_key] = (features, x_train, x_valid, feature_seconds)
            features, x_train, x_valid, feature_seconds = self.cache[feature_key]
            model = make_model(spec["model"], spec["params"], seed)
            print(f"START {stage} seed={seed} {spec}", flush=True)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x_train, self.y[train])
            fit_seconds = time.perf_counter() - start
            train_pred = model.predict(x_train)
            start = time.perf_counter()
            pred = model.predict(x_valid)
            predict_seconds = time.perf_counter() - start
            evaluated_y, evaluated_pred = self.y[valid][included], pred[included]
            if spec["model"] == "MLP":
                parameter_count = sum(a.size for a in model.coefs_ + model.intercepts_)
            elif spec["model"] == "SVM":
                parameter_count = int(model.dual_coef_.size + model.intercept_.size)
            elif spec["model"] == "LR":
                parameter_count = int(model.coef_.size + model.intercept_.size)
            else:
                parameter_count = int(model.feature_log_prob_.size + model.class_log_prior_.size)
            record = dict(
                run_id=key, stage=stage, seed=seed, spec=spec, n_train=len(train), n_valid=int(included.sum()),
                n_raw_valid=len(valid), excluded_validation=valid[~included].tolist(),
                raw_accuracy=accuracy_score(self.y[valid], pred),
                raw_macro_f1=f1_score(self.y[valid], pred, average="macro"),
                n_features=x_train.shape[1], feature_seconds=feature_seconds, fit_seconds=fit_seconds,
                predict_seconds=predict_seconds, accuracy=accuracy_score(evaluated_y, evaluated_pred),
                macro_f1=f1_score(evaluated_y, evaluated_pred, average="macro"),
                train_accuracy=accuracy_score(self.y[train], train_pred),
                train_macro_f1=f1_score(self.y[train], train_pred, average="macro"),
                parameter_count=parameter_count,
                support_vectors=int(len(model.support_)) if hasattr(model, "support_") else None,
                iterations=np.asarray(getattr(model, "n_iter_", [])).tolist(),
                loss_curve=list(getattr(model, "loss_curve_", [])),
                warnings=[str(w.message) for w in caught],
                classification_report=classification_report(evaluated_y, evaluated_pred, output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(evaluated_y, evaluated_pred).tolist(),
            )
            prediction_path = self.output / "validation_predictions" / f"{key}.csv"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(row_id=valid, target=self.y[valid], prediction=pred,
                              included_in_evaluation=included)).to_csv(prediction_path, index=False)
            save_json(record_path, record)
            print(f"DONE {key} F1={record['macro_f1']:.5f} acc={record['accuracy']:.5f} fit={fit_seconds:.1f}s warnings={record['warnings']}", flush=True)
        if key not in self.seen:
            self.records.append(record)
            self.seen.add(key)
        save_json(self.output / "results.json", self.records)
        return record

    @staticmethod
    def best(records):
        return max(records, key=lambda r: (r["macro_f1"], r["accuracy"], -r["fit_seconds"]))

    def execute(self):
        seed = self.config["seed"]
        comparison = []
        for representation in ("bow", "tfidf"):
            for name, grid in self.config["grids"].items():
                for params in grid:
                    comparison.append(self.run(specification(name, params, representation), seed))

        # 词向量实验固定使用 LR，避免把分类器变化混入表示对照。
        for params in self.config["grids"]["LR"]:
            comparison.append(self.run(specification("LR", params, "word2vec"), seed, "representation"))

        tfidf = [r for r in comparison if r["spec"]["representation"] == "tfidf"]
        anchor = self.best(tfidf)
        structural = [anchor]
        for view, weight in [("body", 1.0), ("separate", 0.5), ("separate", 1.0), ("separate", 2.0)]:
            spec = {**anchor["spec"], "view": view, "title_weight": weight}
            structural.append(self.run(spec, seed, "structure"))
        mlp = []
        for hidden in ([64], [128], [64, 32]):
            for alpha in (0.0001, 1.0):
                mlp.append(self.run(specification("MLP", dict(hidden=hidden, alpha=alpha)), seed, "capacity"))

        # 若首轮最优值落在边界，再向外补查一个数量级。
        boundary_checks = []
        for representation in ("bow", "tfidf", "word2vec"):
            for name, parameter in (("NB", "alpha"), ("SVM", "C"), ("LR", "C")):
                if representation == "word2vec" and name != "LR":
                    continue
                group = [r for r in comparison if r["spec"]["model"] == name
                         and r["spec"]["representation"] == representation]
                best = self.best(group)
                values = sorted(r["spec"]["params"][parameter] for r in group)
                value = best["spec"]["params"][parameter]
                if value == values[0]:
                    candidate = value / 10
                elif value == values[-1]:
                    candidate = value * 10
                else:
                    continue
                extra = self.run(specification(name, {parameter: candidate}, representation), seed, "comparison")
                comparison.append(extra)
                boundary_checks.append(dict(model=name, representation=representation,
                                            parameter=parameter, original_best=value, candidate=candidate,
                                            original_macro_f1=best["macro_f1"], extra_run_id=extra["run_id"]))
        save_json(self.output / "boundary_checks.json", boundary_checks)
        tfidf = [r for r in comparison if r["spec"]["representation"] == "tfidf"]

        main_records = [r for r in self.records if r["seed"] == seed]
        selected = self.best(main_records)
        save_json(self.output / "selection.json", dict(
            criterion=f"macro_f1 after excluding train-content overlaps, then accuracy, then shorter classifier fit time on seed {seed}",
            selected_run_id=selected["run_id"], selected_spec=selected["spec"],
            structure_anchor_run_id=anchor["run_id"],
            note="Repeated splits measure sensitivity; they are not an independent test estimate.",
        ))
        repeat_specs = [self.best([r for r in tfidf if r["spec"]["model"] == name])["spec"]
                        for name in self.config["grids"]]
        repeat_specs += [self.best(structural)["spec"], selected["spec"]]
        for repeat_seed in self.config["repeat_seeds"]:
            self.cache.clear()
            for spec in repeat_specs:
                self.run(spec, repeat_seed, "sensitivity")
        return selected

    def predict_final(self, selected):
        spec = selected["spec"]
        features = TextFeatures(self.config, spec["representation"], spec["view"], spec["title_weight"], self.config["seed"])
        x = features.fit_transform(self.texts)
        model = make_model(spec["model"], spec["params"], self.config["seed"])
        start = time.perf_counter()
        model.fit(x, self.y)
        fit_seconds = time.perf_counter() - start
        test_path = self.data_dir / "test_data_unlabeled.csv"
        test = pd.read_csv(test_path, keep_default_na=False)
        if list(test.columns) != ["text"]:
            raise ValueError("Expected one text column in the unlabeled test set")
        predictions = model.predict(features.transform(test.text.to_numpy()))
        if len(predictions) != len(test) or not set(predictions).issubset(set(self.y)):
            raise ValueError("Prediction length or labels are invalid")
        pd.Series(predictions).to_csv(self.output / "prediction.csv", index=False, header=False)
        joblib.dump(dict(features=features, model=model), self.output / "model.joblib", compress=3)
        save_json(self.output / "final_fit.json", dict(
            selected_spec=spec, fit_seconds=fit_seconds, train_rows=len(self.y), test_rows=len(test),
            test_sha256=file_hash(test_path), prediction_sha256=file_hash(self.output / "prediction.csv"),
            label_counts=pd.Series(predictions).value_counts().sort_index().to_dict(),
        ))
        print(f"FINAL {len(predictions)} predictions saved", flush=True)

"""实验入口：运行参数搜索，并按需生成测试集预测。"""

import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from search import Experiment

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data-Project1" / "data-Project1")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    with threadpool_limits(limits=config["threads"]):
        experiment = Experiment(args.data_dir, args.output, config)
        selected = experiment.execute()
        if args.predict:
            experiment.predict_final(selected)


if __name__ == "__main__":
    main()

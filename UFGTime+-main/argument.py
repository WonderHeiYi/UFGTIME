import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Train UFGTime+ with a dataset YAML configuration")
    parser.add_argument("--data_name", default="Covid", help="Dataset key, e.g. ECG or ETTh1_96")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "ufgtime_plus_config.yaml")
    parser.add_argument("--root_path", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=None, help="Override the recorded epoch count")
    parser.add_argument("--window_centering", type=str, default="mean")
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        configs = yaml.safe_load(stream)
    if args.data_name not in configs:
        parser.error(f"Unknown dataset {args.data_name!r}. Choose from: {', '.join(configs)}")
    settings = configs[args.data_name].copy()
    settings.setdefault("scheduler_patience", None)
    patience = settings["scheduler_patience"]
    if patience is not None and (type(patience) is not int or patience < 0):
        parser.error("scheduler_patience must be null or a non-negative integer")
    if args.train_epochs is not None:
        settings["train_epochs"] = args.train_epochs
    if settings["train_epochs"] < 1:
        parser.error("train_epochs must be positive")
    if settings["checkpoint_selection"] not in {"best_val", "last"}:
        parser.error("checkpoint_selection must be best_val or last")
    if settings["long_pred"] not in {0, 1}:
        parser.error("long_pred must be 0 or 1")
    args.run_name = args.data_name
    if args.data_name.startswith("ETT"):
        args.data_name, horizon = args.data_name.rsplit("_", 1)
        if settings["pred_len"] != int(horizon):
            parser.error("The dataset suffix and configured pred_len must agree")
    vars(args).update(settings)
    args.root_path = args.root_path.resolve()
    return args

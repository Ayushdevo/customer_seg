from __future__ import annotations

import argparse

from .predict import CLUSTER_LABELS, predict_customer_cluster
from .loaders import LoaderError, load_models


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer segmentation CLI")
    parser.add_argument("--recency", type=float, required=True, help="Days since the customer's last purchase")
    parser.add_argument("--frequency", type=float, required=True, help="Total number of purchases")
    parser.add_argument("--monetary", type=float, required=True, help="Total spend in USD")
    parser.add_argument("--model", default="models/kmeans_model.pkl", help="Trusted fitted model artifact")
    parser.add_argument("--scaler", default="models/rfm_scaler.pkl", help="Trusted fitted RFM scaler")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        model, scaler = load_models(args.model, args.scaler)
        cluster = predict_customer_cluster(model, scaler, args.recency, args.frequency, args.monetary)
    except (LoaderError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Cluster {cluster}: {CLUSTER_LABELS.get(cluster, 'Unknown segment')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())

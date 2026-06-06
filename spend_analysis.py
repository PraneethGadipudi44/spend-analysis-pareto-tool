"""
Spend analysis and Pareto supplier classification tool.

Dataset used:
Fulton County vendor payments open data (Socrata)
https://data.fultoncountyga.gov/Tax-Finance/Vendor-Payments/mxhc-krcg

The script downloads a public spend dataset with supplier, category, and amount fields,
cleans it, standardizes supplier names, classifies suppliers into A/B/C Pareto groups,
exports the classified supplier file, and saves a Pareto chart image.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


DATASET_URL = (
    "https://data.fultoncountyga.gov/resource/mxhc-krcg.csv"
    "?$select=fiscal_year,vendor_legal_name,object_name,amount"
    "&$limit=500000"
)

CLASS_COLORS = {"A": "#d73027", "B": "#fdae61", "C": "#74add1"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run spend analysis and Pareto classification on a public spend dataset."
    )
    parser.add_argument(
        "--dataset-url",
        default=DATASET_URL,
        help="Public CSV dataset URL to load.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=script_dir / "data" / "fulton_vendor_payments.csv",
        help="Local cache file for the raw public dataset.",
    )
    parser.add_argument(
        "--classified-output",
        type=Path,
        default=script_dir / "supplier_pareto_classification.csv",
        help="Output CSV for classified supplier spend.",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=script_dir / "pareto_chart.png",
        help="Output PNG for the Pareto chart.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force a fresh download of the dataset.",
    )
    return parser.parse_args()


def load_dataset(dataset_url: str, cache_path: Path, refresh_data: bool) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh_data or not cache_path.exists():
        print(f"Downloading dataset from {dataset_url}")
        df = pd.read_csv(dataset_url)
        df.to_csv(cache_path, index=False)
    else:
        print(f"Using cached dataset at {cache_path}")
        df = pd.read_csv(cache_path)
    return df


def standardize_supplier_name(name: object) -> str:
    if pd.isna(name):
        return "UNKNOWN SUPPLIER"

    value = str(name).upper().strip()
    value = value.replace("&", " AND ")
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)
    value = re.sub(r"\b(THE)\b", " ", value)
    value = re.sub(
        r"\b(INC|INCORPORATED|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC|PC)\b",
        " ",
        value,
    )
    value = re.sub(r"\s+", " ", value).strip()
    return value or "UNKNOWN SUPPLIER"


def cleanse_dataset(df: pd.DataFrame) -> pd.DataFrame:
    working = df.rename(
        columns={
            "vendor_legal_name": "supplier_name",
            "object_name": "category",
            "amount": "spend_amount",
        }
    ).copy()

    working["supplier_name"] = (
        working["supplier_name"].fillna("Unknown Supplier").astype(str).str.strip()
    )
    working["category"] = (
        working["category"].fillna("Unspecified Category").astype(str).str.strip()
    )
    working["spend_amount"] = pd.to_numeric(working["spend_amount"], errors="coerce")
    working = working.dropna(subset=["spend_amount"])
    working = working.loc[working["spend_amount"] > 0].copy()
    working["supplier_name_standardized"] = working["supplier_name"].map(standardize_supplier_name)
    return working


def choose_display_supplier_name(df: pd.DataFrame) -> pd.DataFrame:
    display = (
        df.groupby(["supplier_name_standardized", "supplier_name"], as_index=False)["spend_amount"]
        .sum()
        .sort_values(
            ["supplier_name_standardized", "spend_amount", "supplier_name"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["supplier_name_standardized"])
        .rename(columns={"supplier_name": "display_supplier_name"})
        [["supplier_name_standardized", "display_supplier_name"]]
    )
    return display


def choose_primary_category(df: pd.DataFrame) -> pd.DataFrame:
    primary = (
        df.groupby(["supplier_name_standardized", "category"], as_index=False)["spend_amount"]
        .sum()
        .sort_values(
            ["supplier_name_standardized", "spend_amount", "category"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["supplier_name_standardized"])
        .rename(columns={"category": "primary_category"})
        [["supplier_name_standardized", "primary_category"]]
    )
    return primary


def classify_suppliers(df: pd.DataFrame) -> pd.DataFrame:
    supplier_spend = (
        df.groupby("supplier_name_standardized", as_index=False)
        .agg(total_spend=("spend_amount", "sum"), transaction_count=("spend_amount", "size"))
        .sort_values("total_spend", ascending=False)
        .reset_index(drop=True)
    )

    supplier_spend["supplier_rank"] = np.arange(1, len(supplier_spend) + 1)
    supplier_spend["spend_share_pct"] = (
        supplier_spend["total_spend"] / supplier_spend["total_spend"].sum() * 100
    )
    supplier_spend["cumulative_spend_pct"] = supplier_spend["spend_share_pct"].cumsum()

    supplier_spend["pareto_class"] = np.select(
        [
            supplier_spend["cumulative_spend_pct"] <= 80,
            supplier_spend["cumulative_spend_pct"] <= 95,
        ],
        ["A", "B"],
        default="C",
    )

    display_names = choose_display_supplier_name(df)
    primary_categories = choose_primary_category(df)

    classified = (
        supplier_spend.merge(display_names, on="supplier_name_standardized", how="left")
        .merge(primary_categories, on="supplier_name_standardized", how="left")
        .loc[
            :,
            [
                "supplier_rank",
                "display_supplier_name",
                "supplier_name_standardized",
                "primary_category",
                "transaction_count",
                "total_spend",
                "spend_share_pct",
                "cumulative_spend_pct",
                "pareto_class",
            ],
        ]
    )

    return classified


def class_summary(classified: pd.DataFrame) -> pd.DataFrame:
    summary = (
        classified.groupby("pareto_class", as_index=False)
        .agg(
            supplier_count=("display_supplier_name", "size"),
            class_spend=("total_spend", "sum"),
        )
        .sort_values("pareto_class")
    )
    summary["class_spend_pct"] = summary["class_spend"] / summary["class_spend"].sum() * 100
    return summary


def create_pareto_chart(classified: pd.DataFrame, chart_output: Path) -> None:
    chart_output.parent.mkdir(parents=True, exist_ok=True)

    x = classified["supplier_rank"].to_numpy()
    spend_millions = classified["total_spend"].to_numpy() / 1_000_000
    cumulative_pct = classified["cumulative_spend_pct"].to_numpy()
    bar_colors = classified["pareto_class"].map(CLASS_COLORS).to_list()

    a_cutoff = int((classified["pareto_class"] == "A").sum())
    b_cutoff = int(classified["pareto_class"].isin(["A", "B"]).sum())

    summary = class_summary(classified).set_index("pareto_class")

    fig, ax1 = plt.subplots(figsize=(15, 8))
    ax1.bar(x, spend_millions, color=bar_colors, width=1.0, edgecolor="none")
    ax1.set_xlabel("Supplier rank by total spend")
    ax1.set_ylabel("Supplier spend (millions)")
    ax1.set_title("Supplier Spend Pareto Analysis (A/B/C Classification)")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_pct, color="black", linewidth=2.2, label="Cumulative spend %")
    ax2.set_ylabel("Cumulative spend (%)")
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color=CLASS_COLORS["A"], linestyle="--", linewidth=1.2)
    ax2.axhline(95, color=CLASS_COLORS["B"], linestyle="--", linewidth=1.2)
    ax2.axhline(100, color=CLASS_COLORS["C"], linestyle=":", linewidth=1.0)

    for cutoff, label in [(a_cutoff, "A"), (b_cutoff, "B")]:
        ax1.axvline(cutoff, color="#555555", linestyle=":", linewidth=1.2)
        ax2.text(
            cutoff,
            101,
            f"{label} cutoff",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=9,
            color="#444444",
        )

    legend_items = [
        Patch(facecolor=CLASS_COLORS["A"], label="A suppliers (top cumulative 80%)"),
        Patch(facecolor=CLASS_COLORS["B"], label="B suppliers (next cumulative 15%)"),
        Patch(facecolor=CLASS_COLORS["C"], label="C suppliers (remaining spend)"),
    ]
    ax1.legend(handles=legend_items, loc="upper right")

    summary_lines = []
    for label in ["A", "B", "C"]:
        row = summary.loc[label]
        summary_lines.append(
            f"{label}: {int(row['supplier_count'])} suppliers | "
            f"{row['class_spend_pct']:.1f}% of spend"
        )

    fig.text(
        0.12,
        0.78,
        "Class summary\n" + "\n".join(summary_lines),
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(chart_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    raw = load_dataset(args.dataset_url, args.cache_path, args.refresh_data)
    clean = cleanse_dataset(raw)
    classified = classify_suppliers(clean)

    args.classified_output.parent.mkdir(parents=True, exist_ok=True)
    classified.to_csv(args.classified_output, index=False)
    create_pareto_chart(classified, args.chart_output)

    summary = class_summary(classified)

    print("\nSpend analysis complete.\n")
    print(f"Transactions analysed: {len(clean):,}")
    print(f"Suppliers classified: {len(classified):,}")
    print("\nPareto class summary:")
    print(summary.to_string(index=False))
    print(f"\nClassified supplier file: {args.classified_output}")
    print(f"Pareto chart: {args.chart_output}")


if __name__ == "__main__":
    main()

# Spend Analysis and Pareto Classification Tool

This project is a Python-based spend analysis tool built on top of a public procurement payments dataset.

The goal is to load supplier payment data, clean it, standardize supplier names, calculate supplier-level spend, classify suppliers into A, B, and C groups using Pareto logic, and generate a clear chart showing how spend is distributed.

## Dataset

This project uses a public vendor payments dataset from Fulton County open data.

Source:
https://data.fultoncountyga.gov/Tax-Finance/Vendor-Payments/mxhc-krcg

The dataset includes fields that work well for this task:

- supplier name
- spend category
- payment amount

In the script, these fields are mapped from:

- `vendor_legal_name`
- `object_name`
- `amount`

## What the Script Does

The script:

- loads the public dataset
- caches a local copy for repeat runs
- cleans missing values
- removes invalid or non-positive spend amounts
- standardizes supplier names
- aggregates spend at the supplier level
- calculates cumulative spend contribution
- classifies suppliers into:
  - A suppliers: top cumulative 80% of spend
  - B suppliers: next cumulative 15% of spend
  - C suppliers: remaining spend
- exports the classified supplier file
- generates a Pareto chart image

## Files in This Repository

- `spend_analysis.py` — main Python script
- `pareto_chart.png` — generated Pareto chart
- `supplier_pareto_classification.csv` — exported classified supplier spend file

## Pareto Classification Logic

The tool follows a standard Pareto-style supplier classification:

- **A suppliers** represent the suppliers contributing to roughly the first 80% of cumulative spend
- **B suppliers** represent the next 15% of cumulative spend
- **C suppliers** represent the remaining suppliers

This helps identify which suppliers matter most from a strategic sourcing and spend concentration point of view.

## How to Run

Install the required packages:

```bash
pip install pandas numpy matplotlib

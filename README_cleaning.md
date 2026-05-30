# Data Cleaning & Reporting Automation

Automates the full data cleaning pipeline and generates visual reports from raw, messy sales data.

## What It Does

| Phase | Task |
|---|---|
| Phase 1 | Loads raw CSV and audits all data quality issues |
| Phase 2 | Cleans: missing values, duplicates, outliers, bad formats, negative prices |
| Phase 3 | Computes KPIs and summary statistics |
| Phase 4 | Generates 7 professional charts |
| Phase 5 | Creates a formatted multi-sheet Excel report |
| Phase 6 | Exports cleaned CSV and cleaning log |

## Issues Detected & Fixed

- Missing values → filled with median / mode
- Duplicate rows → removed
- Negative/zero prices → rows dropped
- Inconsistent category names (e.g. `electronics`, `Electroncs`) → standardised
- Mixed date formats (`2024-01-15` and `15/01/2024`) → unified
- Revenue mismatches → recalculated as qty × unit_price
- Outliers → removed via IQR method

## Project Structure

```
data-cleaning-automation/
├── dirty_sales_data.csv           # Raw input (1000 rows with issues)
├── data_cleaning_automation.py    # Main automation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## How to Run

```bash
pip install -r requirements.txt
python data_cleaning_automation.py
```

## Output Files Generated (auto-created, not pushed)

| File | Description |
|---|---|
| cleaned_sales_data.csv | Fully cleaned dataset (893 rows) |
| cleaning_log.csv | Log of every issue found and fixed |
| automated_report.xlsx | 5-sheet Excel report with formatting |
| 01_data_quality_audit.png | Issues found + data retention pie |
| 02_monthly_revenue.png | Revenue trend over time |
| 03_revenue_breakdown.png | Revenue by category and region |
| 04_top_products.png | Top products by revenue |
| 05_salesperson_performance.png | Sales team comparison |
| 06_quarterly_heatmap.png | Revenue heatmap by quarter/year |
| 07_before_after_distribution.png | Price distribution before vs after |

## Results

- 1000 rows input → 893 rows clean (89.3% retained)
- 107 rows removed (duplicates, outliers, bad prices)
- Total revenue: $20,245,125
- Top product: Standing Desk

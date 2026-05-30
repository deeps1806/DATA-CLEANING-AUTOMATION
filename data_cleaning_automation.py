# ══════════════════════════════════════════════════════════════
#   DATA CLEANING & REPORTING AUTOMATION
#   Handles: missing values, duplicates, inconsistent data,
#            outliers, format errors → generates full report
# ══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
                     'axes.grid':True,'grid.alpha':0.25,'font.size':11})

print("=" * 65)
print("   DATA CLEANING & REPORTING AUTOMATION")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
#   PHASE 1 — LOAD & AUDIT RAW DATA
# ══════════════════════════════════════════════════════════════
print("\n📂  PHASE 1 — Loading & Auditing Raw Data")
print("-" * 45)

df_raw = pd.read_csv('dirty_sales_data.csv', dtype=str)
total_raw = len(df_raw)
print(f"  Rows loaded     : {total_raw}")
print(f"  Columns         : {list(df_raw.columns)}")

# Audit log — track every issue found
audit = {
    'missing_values'    : {},
    'duplicates'        : 0,
    'negative_prices'   : 0,
    'bad_date_formats'  : 0,
    'inconsistent_cats' : 0,
    'whitespace_fixed'  : 0,
    'outliers_removed'  : 0,
    'revenue_recalculated': 0,
}

# Count missing per column
for col in df_raw.columns:
    missing = df_raw[col].isnull().sum() + (df_raw[col].str.strip() == '').sum()
    if missing: audit['missing_values'][col] = int(missing)

print(f"\n  Missing value audit:")
for col, cnt in audit['missing_values'].items():
    print(f"    {col:20s} : {cnt} missing")

# ══════════════════════════════════════════════════════════════
#   PHASE 2 — DATA CLEANING (step by step)
# ══════════════════════════════════════════════════════════════
print("\n🧹  PHASE 2 — Cleaning Data")
print("-" * 45)

df = df_raw.copy()

# ── 2a: Strip whitespace from all string columns ───────────
ws_before = df.apply(lambda c: (c != c.str.strip()).sum() if c.dtype==object else 0).sum()
df = df.apply(lambda c: c.str.strip() if c.dtype == object else c)
audit['whitespace_fixed'] = int(ws_before)
print(f"  ✔ Whitespace stripped      : {ws_before} cells fixed")

# ── 2b: Fix date formats ───────────────────────────────────
def parse_date(d):
    for fmt in ('%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%d-%m-%Y'):
        try: return pd.to_datetime(d, format=fmt)
        except: pass
    return pd.NaT

before_nat = df['date'].isnull().sum()
df['date'] = df['date'].apply(parse_date)
after_nat  = df['date'].isnull().sum()
audit['bad_date_formats'] = int(after_nat - before_nat)
print(f"  ✔ Date formats standardised : {df['date'].notna().sum()} valid  |  {after_nat} unparseable → dropped")
df.dropna(subset=['date'], inplace=True)

# ── 2c: Standardise category ──────────────────────────────
cat_map = {
    'electronics':'Electronics','electroncs':'Electronics','electronic':'Electronics',
    'furniture':'Furniture','wearables':'Wearables','wearable':'Wearables',
    'accessories':'Accessories','accessory':'Accessories',
}
before_cats = df['category'].copy()
df['category'] = df['category'].str.strip().str.lower().map(
    lambda x: cat_map.get(x, x.title()) if isinstance(x,str) else x)
audit['inconsistent_cats'] = int((before_cats.str.lower() != df['category'].str.lower()).sum())
print(f"  ✔ Categories standardised   : {audit['inconsistent_cats']} values fixed")

# ── 2d: Standardise region ────────────────────────────────
df['region'] = df['region'].str.strip().str.title()

# ── 2e: Convert numeric columns ───────────────────────────
df['quantity']   = pd.to_numeric(df['quantity'],   errors='coerce')
df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
df['revenue']    = pd.to_numeric(df['revenue'],    errors='coerce')

# ── 2f: Fill missing quantity & price with median ─────────
df['quantity'].fillna(df['quantity'].median(),     inplace=True)
df['unit_price'].fillna(df['unit_price'].median(), inplace=True)
df['quantity']   = df['quantity'].round(0).astype('Int64')

# Fill missing region with mode
df['region'].replace('', np.nan, inplace=True)
df['region'].fillna(df['region'].mode()[0], inplace=True)

# Fill missing salesperson with 'Unknown'
df['salesperson'].replace('', np.nan, inplace=True)
df['salesperson'].fillna('Unknown', inplace=True)

print(f"  ✔ Missing numerics filled   : median imputation applied")

# ── 2g: Remove negative / zero prices ─────────────────────
neg_mask = df['unit_price'] <= 0
audit['negative_prices'] = int(neg_mask.sum())
df = df[~neg_mask].copy()
print(f"  ✔ Negative prices removed   : {audit['negative_prices']} rows dropped")

# ── 2h: Recalculate revenue ───────────────────────────────
expected_rev = df['quantity'] * df['unit_price']
mismatch = (abs(df['revenue'] - expected_rev) / expected_rev.clip(lower=1) > 0.5).sum()
df['revenue'] = expected_rev.round(2)
audit['revenue_recalculated'] = int(mismatch)
print(f"  ✔ Revenue recalculated      : {mismatch} mismatches corrected")

# ── 2i: Remove duplicates ─────────────────────────────────
dup_cols = ['customer_id','date','product','quantity','unit_price']
before_dup = len(df)
df.drop_duplicates(subset=dup_cols, inplace=True)
audit['duplicates'] = before_dup - len(df)
print(f"  ✔ Duplicates removed        : {audit['duplicates']} rows dropped")

# ── 2j: Remove revenue outliers (IQR method) ──────────────
Q1, Q3 = df['revenue'].quantile(0.25), df['revenue'].quantile(0.75)
IQR = Q3 - Q1
before_out = len(df)
df = df[(df['revenue'] >= Q1-1.5*IQR) & (df['revenue'] <= Q3+1.5*IQR)].copy()
audit['outliers_removed'] = before_out - len(df)
print(f"  ✔ Outliers removed          : {audit['outliers_removed']} rows dropped")

# ── 2k: Add derived columns ───────────────────────────────
df['year']    = df['date'].dt.year
df['month']   = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['quarter'] = df['date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
df.reset_index(drop=True, inplace=True)

total_clean = len(df)
rows_removed = total_raw - total_clean
print(f"\n  Before cleaning : {total_raw} rows")
print(f"  After cleaning  : {total_clean} rows  ({rows_removed} removed  |  {100*total_clean/total_raw:.1f}% retained)")

# ══════════════════════════════════════════════════════════════
#   PHASE 3 — SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════
print("\n📊  PHASE 3 — Summary Statistics")
print("-" * 45)

total_rev   = df['revenue'].sum()
avg_order   = df['revenue'].mean()
total_orders= len(df)
top_product = df.groupby('product')['revenue'].sum().idxmax()
top_region  = df.groupby('region')['revenue'].sum().idxmax()
top_sales   = df.groupby('salesperson')['revenue'].sum().idxmax()
date_range  = f"{df['date'].min().date()} → {df['date'].max().date()}"

print(f"  Total revenue   : ${total_rev:,.2f}")
print(f"  Total orders    : {total_orders}")
print(f"  Avg order value : ${avg_order:,.2f}")
print(f"  Top product     : {top_product}")
print(f"  Top region      : {top_region}")
print(f"  Top salesperson : {top_sales}")
print(f"  Date range      : {date_range}")

# ══════════════════════════════════════════════════════════════
#   PHASE 4 — CHARTS
# ══════════════════════════════════════════════════════════════
print("\n🎨  PHASE 4 — Generating Charts")
print("-" * 45)

COLORS = ['#378ADD','#1D9E75','#D85A30','#7F77DD','#BA7517','#D4537E','#639922','#5DCAA5']

# ── Chart 1: Data Quality Audit (Before vs After) ──────────
issues = {
    'Missing\nValues':     sum(audit['missing_values'].values()),
    'Duplicates':          audit['duplicates'],
    'Negative\nPrices':    audit['negative_prices'],
    'Date Format\nErrors': audit['bad_date_formats'],
    'Category\nInconsist.':audit['inconsistent_cats'],
    'Outliers':            audit['outliers_removed'],
    'Whitespace':          audit['whitespace_fixed'],
}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(issues.keys(), issues.values(),
                   color=[COLORS[i%len(COLORS)] for i in range(len(issues))],
                   edgecolor='white', width=0.6)
axes[0].set_title('Data Quality Issues Found', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
for bar, val in zip(bars, issues.values()):
    if val > 0:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='500')
axes[0].tick_params(axis='x', labelsize=9)

labels_pie = ['Retained\n(Clean)', 'Removed\n(Issues)']
sizes_pie  = [total_clean, rows_removed]
axes[1].pie(sizes_pie, labels=labels_pie, colors=['#1D9E75','#D85A30'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2}, textprops={'fontsize':12})
axes[1].set_title(f'Data Retention\n({total_raw} → {total_clean} rows)', fontsize=13, fontweight='bold')
plt.suptitle('Data Quality Audit Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_data_quality_audit.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_data_quality_audit.png")

# ── Chart 2: Monthly Revenue Trend ────────────────────────
monthly = df.groupby(['year','month'])['revenue'].sum().reset_index()
monthly['period'] = pd.to_datetime(monthly[['year','month']].assign(day=1))
monthly.sort_values('period', inplace=True)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly['period'], monthly['revenue'], color='#378ADD',
        linewidth=2.5, marker='o', markersize=5, label='Monthly Revenue')
ax.fill_between(monthly['period'], monthly['revenue'], alpha=0.12, color='#378ADD')
ax.set_title('Monthly Revenue Trend (Cleaned Data)', fontsize=14, fontweight='bold')
ax.set_ylabel('Revenue ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.xticks(rotation=30)
plt.legend(); plt.tight_layout()
plt.savefig('02_monthly_revenue.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_monthly_revenue.png")

# ── Chart 3: Revenue by Category & Region ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cat_rev = df.groupby('category')['revenue'].sum().sort_values()
axes[0].barh(cat_rev.index, cat_rev.values,
             color=COLORS[:len(cat_rev)], edgecolor='white', height=0.6)
axes[0].set_title('Revenue by Category', fontweight='bold')
axes[0].set_xlabel('Revenue ($)')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
for i,(v,n) in enumerate(zip(cat_rev.values, cat_rev.index)):
    axes[0].text(v+500, i, f'${v/1000:.0f}k', va='center', fontsize=10)

reg_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
axes[1].bar(reg_rev.index, reg_rev.values,
            color=COLORS[4:4+len(reg_rev)], edgecolor='white', width=0.6)
axes[1].set_title('Revenue by Region', fontweight='bold')
axes[1].set_ylabel('Revenue ($)')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
plt.suptitle('Revenue Breakdown', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_revenue_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_revenue_breakdown.png")

# ── Chart 4: Top Products ─────────────────────────────────
top_prod = df.groupby('product')['revenue'].sum().sort_values(ascending=True).tail(8)
fig, ax = plt.subplots(figsize=(12, 6))
colors_p = ['#D85A30' if v==top_prod.max() else '#378ADD' for v in top_prod.values]
ax.barh(top_prod.index, top_prod.values, color=colors_p, edgecolor='white', height=0.65)
ax.set_title('Top Products by Revenue', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Revenue ($)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
for i,(v,n) in enumerate(zip(top_prod.values, top_prod.index)):
    ax.text(v+500, i, f'${v/1000:.0f}k', va='center', fontsize=10, fontweight='500')
plt.tight_layout()
plt.savefig('04_top_products.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_top_products.png")

# ── Chart 5: Salesperson Performance ─────────────────────
sp_perf = df.groupby('salesperson').agg(
    revenue=('revenue','sum'), orders=('order_id','count')).sort_values('revenue',ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(len(sp_perf)), sp_perf['revenue'],
            color=[COLORS[i%len(COLORS)] for i in range(len(sp_perf))], edgecolor='white', width=0.65)
axes[0].set_xticks(range(len(sp_perf)))
axes[0].set_xticklabels([n.split()[0] for n in sp_perf.index], rotation=35, ha='right')
axes[0].set_title('Revenue per Salesperson', fontweight='bold')
axes[0].set_ylabel('Revenue ($)')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x/1000:.0f}k'))
axes[1].bar(range(len(sp_perf)), sp_perf['orders'],
            color=[COLORS[i%len(COLORS)] for i in range(len(sp_perf))], edgecolor='white', width=0.65)
axes[1].set_xticks(range(len(sp_perf)))
axes[1].set_xticklabels([n.split()[0] for n in sp_perf.index], rotation=35, ha='right')
axes[1].set_title('Orders per Salesperson', fontweight='bold')
axes[1].set_ylabel('Number of Orders')
plt.suptitle('Salesperson Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_salesperson_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_salesperson_performance.png")

# ── Chart 6: Quarterly Revenue Heatmap ────────────────────
df['revenue_float'] = df['revenue'].astype(float)
heat_data = df.pivot_table(values='revenue_float', index='quarter',
                            columns='year', aggfunc='sum')
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heat_data, annot=True, fmt=',.0f', cmap='Blues',
            linewidths=0.5, ax=ax, cbar_kws={'label':'Revenue ($)'})
ax.set_title('Quarterly Revenue Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('06_quarterly_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_quarterly_heatmap.png")

# ── Chart 7: Before vs After — Price Distribution ─────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
raw_prices = pd.to_numeric(df_raw['unit_price'], errors='coerce').dropna()
axes[0].hist(raw_prices, bins=40, color='#D85A30', edgecolor='white', alpha=0.85)
axes[0].set_title('Unit Price — BEFORE Cleaning', fontweight='bold')
axes[0].set_xlabel('Price ($)'); axes[0].set_ylabel('Frequency')
axes[1].hist(df['unit_price'], bins=40, color='#1D9E75', edgecolor='white', alpha=0.85)
axes[1].set_title('Unit Price — AFTER Cleaning', fontweight='bold')
axes[1].set_xlabel('Price ($)'); axes[1].set_ylabel('Frequency')
plt.suptitle('Data Distribution: Before vs After Cleaning', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('07_before_after_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_before_after_distribution.png")

# ══════════════════════════════════════════════════════════════
#   PHASE 5 — AUTOMATED EXCEL REPORT
# ══════════════════════════════════════════════════════════════
print("\n📝  PHASE 5 — Generating Excel Report")
print("-" * 45)

report_path = 'automated_report.xlsx'
with pd.ExcelWriter(report_path, engine='xlsxwriter') as writer:
    wb  = writer.book

    # Formats
    hdr_fmt  = wb.add_format({'bold':True,'bg_color':'#378ADD','font_color':'white','border':1,'align':'center','valign':'vcenter','font_size':12})
    sub_fmt  = wb.add_format({'bold':True,'bg_color':'#E8F2FC','border':1,'font_size':11})
    num_fmt  = wb.add_format({'num_format':'$#,##0.00','border':1})
    int_fmt  = wb.add_format({'num_format':'#,##0','border':1})
    pct_fmt  = wb.add_format({'num_format':'0.0%','border':1})
    cell_fmt = wb.add_format({'border':1,'font_size':10})
    title_fmt= wb.add_format({'bold':True,'font_size':16,'font_color':'#378ADD'})
    good_fmt = wb.add_format({'bg_color':'#E1F5EE','border':1,'bold':True,'font_color':'#0F6E56'})
    warn_fmt = wb.add_format({'bg_color':'#FAECE7','border':1,'bold':True,'font_color':'#993C1D'})

    # ── Sheet 1: Executive Summary ─────────────────────────
    ws = wb.add_worksheet('Executive Summary')
    ws.set_column('A:A', 30); ws.set_column('B:B', 22)
    ws.write('A1', '📊 DATA CLEANING & REPORTING AUTOMATION', title_fmt)
    ws.write('A2', f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', wb.add_format({'italic':True,'font_color':'#666'}))
    ws.write('A4', 'KPI', hdr_fmt); ws.write('B4', 'Value', hdr_fmt)
    kpis = [
        ('Total Revenue',        f'${total_rev:,.2f}'),
        ('Total Orders (clean)', f'{total_orders:,}'),
        ('Avg Order Value',      f'${avg_order:,.2f}'),
        ('Top Product',          top_product),
        ('Top Region',           top_region),
        ('Top Salesperson',      top_sales),
        ('Date Range',           date_range),
        ('Rows Before Cleaning', f'{total_raw:,}'),
        ('Rows After Cleaning',  f'{total_clean:,}'),
        ('Rows Removed',         f'{rows_removed:,}'),
        ('Data Retention Rate',  f'{100*total_clean/total_raw:.1f}%'),
    ]
    for r,(k,v) in enumerate(kpis, start=4):
        ws.write(r, 0, k, sub_fmt)
        ws.write(r, 1, v, cell_fmt)

    # ── Sheet 2: Cleaning Log ──────────────────────────────
    ws2 = wb.add_worksheet('Cleaning Log')
    ws2.set_column('A:A', 30); ws2.set_column('B:C', 18)
    ws2.write('A1', 'Cleaning Log', title_fmt)
    headers = ['Issue Type', 'Count', 'Action Taken']
    for c,h in enumerate(headers): ws2.write(1, c, h, hdr_fmt)
    log_rows = [
        ('Missing values (quantity)',   audit['missing_values'].get('quantity',0),   'Filled with median'),
        ('Missing values (unit_price)', audit['missing_values'].get('unit_price',0), 'Filled with median'),
        ('Missing values (region)',     audit['missing_values'].get('region',0),     'Filled with mode'),
        ('Missing values (salesperson)',audit['missing_values'].get('salesperson',0),'Filled with "Unknown"'),
        ('Whitespace in cells',         audit['whitespace_fixed'],                   'Stripped'),
        ('Inconsistent categories',     audit['inconsistent_cats'],                  'Standardised to title case'),
        ('Bad date formats',            audit['bad_date_formats'],                   'Parsed / dropped if unparseable'),
        ('Negative/zero unit prices',   audit['negative_prices'],                    'Rows removed'),
        ('Duplicate rows',              audit['duplicates'],                          'Removed (kept first)'),
        ('Revenue outliers',            audit['outliers_removed'],                    'Removed via IQR method'),
        ('Revenue mismatches',          audit['revenue_recalculated'],               'Recalculated: qty × price'),
    ]
    for r,(issue, cnt, action) in enumerate(log_rows, start=2):
        fmt = warn_fmt if cnt > 0 else good_fmt
        ws2.write(r, 0, issue,  fmt)
        ws2.write(r, 1, cnt,    fmt)
        ws2.write(r, 2, action, cell_fmt)

    # ── Sheet 3: Cleaned Data ──────────────────────────────
    out_cols = ['order_id','customer_id','date','product','category',
                'region','quantity','unit_price','revenue','salesperson','year','quarter']
    df[out_cols].to_excel(writer, sheet_name='Cleaned Data', index=False)
    ws3 = writer.sheets['Cleaned Data']
    for c, col in enumerate(out_cols):
        ws3.write(0, c, col, hdr_fmt)
    ws3.set_column('A:A', 10); ws3.set_column('B:B', 14)
    ws3.set_column('C:C', 14); ws3.set_column('D:E', 20)
    ws3.set_column('F:F', 10); ws3.set_column('G:H', 12)
    ws3.set_column('I:I', 14); ws3.set_column('J:J', 18)

    # ── Sheet 4: Monthly Summary ───────────────────────────
    monthly_s = df.groupby(['year','quarter','month','month_name']).agg(
        orders=('order_id','count'), revenue=('revenue','sum'),
        avg_order=('revenue','mean'), units=('quantity','sum')).reset_index()
    monthly_s.sort_values(['year','month'], inplace=True)
    monthly_s.to_excel(writer, sheet_name='Monthly Summary', index=False)
    ws4 = writer.sheets['Monthly Summary']
    for c, col in enumerate(monthly_s.columns):
        ws4.write(0, c, col, hdr_fmt)

    # ── Sheet 5: Product Summary ───────────────────────────
    prod_s = df.groupby(['product','category']).agg(
        orders=('order_id','count'), total_revenue=('revenue','sum'),
        avg_price=('unit_price','mean'), total_units=('quantity','sum')).reset_index()
    prod_s.sort_values('total_revenue', ascending=False, inplace=True)
    prod_s.to_excel(writer, sheet_name='Product Summary', index=False)
    ws5 = writer.sheets['Product Summary']
    for c, col in enumerate(prod_s.columns):
        ws5.write(0, c, col, hdr_fmt)
    ws5.set_column('A:B', 22); ws5.set_column('C:F', 16)

print(f"  Saved: {report_path}")

# ══════════════════════════════════════════════════════════════
#   PHASE 6 — SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════
print("\n💾  PHASE 6 — Saving CSV Outputs")
print("-" * 45)

df[out_cols].to_csv('cleaned_sales_data.csv', index=False)
print("  Saved: cleaned_sales_data.csv")

cleaning_log = pd.DataFrame(log_rows, columns=['Issue','Count','Action'])
cleaning_log.to_csv('cleaning_log.csv', index=False)
print("  Saved: cleaning_log.csv")

print("\n" + "=" * 65)
print("  ✅  ALL DONE")
print("  Files ready to push to GitHub:")
print("    dirty_sales_data.csv          ← raw input data")
print("    data_cleaning_automation.py   ← main script")
print("    requirements.txt              ← dependencies")
print("    README.md                     ← documentation")
print("  Auto-generated (DO NOT push):")
print("    cleaned_sales_data.csv")
print("    cleaning_log.csv")
print("    automated_report.xlsx")
print("    01–07 chart PNGs")
print("=" * 65)
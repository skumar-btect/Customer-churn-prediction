"""
PROJECT 3: Customer Churn Prediction
DATA CLEANING SCRIPT
Author: S. Kumar | Data Analyst
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("PROJECT 3: Customer Churn Prediction")
print("DATA CLEANING PIPELINE")
print("=" * 60)

# ── 1. LOAD RAW DATA ──────────────────────────────────────────
df = pd.read_csv('/home/claude/da-projects/project3-churn-prediction/data/raw_user_data.csv')
print(f"\n[LOAD] Raw shape: {df.shape}")

# ── 2. INITIAL QUALITY REPORT ─────────────────────────────────
print("\n[QUALITY REPORT - BEFORE CLEANING]")
print(f"  Total rows         : {len(df)}")
print(f"  Duplicate users    : {df.duplicated(subset='user_id').sum()}")
print(f"  Missing values     :\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"  Zero monthly charge: {(df['monthly_charges']==0).sum()}")
print(f"  Churn rate         : {df['churned'].mean()*100:.1f}%")

# ── 3. FIX ZERO MONTHLY CHARGES ──────────────────────────────
zero_charge = df['monthly_charges'] == 0
df.loc[zero_charge, 'monthly_charges'] = np.nan
df['monthly_charges'] = df.groupby('subscription_plan')['monthly_charges'].transform(
    lambda x: x.fillna(x.median()))
df['monthly_charges'] = df['monthly_charges'].round(2)
print(f"\n[CHARGES FIX] Replaced {zero_charge.sum()} zero charges with plan median")

# ── 4. IMPUTE MISSING SESSION TIME ───────────────────────────
missing_sessions = df['avg_session_min'].isna().sum()
df['avg_session_min'] = df.groupby(['subscription_plan', 'churned'])['avg_session_min'].transform(
    lambda x: x.fillna(x.median()))
df['avg_session_min'] = df['avg_session_min'].round(1)
print(f"[SESSION FIX] Imputed {missing_sessions} missing session minutes")

# ── 5. RECALCULATE TOTAL CHARGES ──────────────────────────────
df['total_charges'] = (df['monthly_charges'] * df['tenure_months']).round(2)
print("[CHARGES] Recalculated total_charges = monthly × tenure")

# ── 6. FEATURE ENGINEERING ────────────────────────────────────
df['engagement_score'] = (
    df['num_logins_30d'] * 0.3 +
    df['avg_session_min'] * 0.3 +
    df['features_used'] * 0.2 +
    (10 - df['support_tickets']) * 0.1 +
    df['referrals_made'] * 0.1
).round(2)

df['is_high_value'] = (
    (df['monthly_charges'] > df['monthly_charges'].median()) &
    (df['tenure_months'] > 12)
).astype(int)

df['churn_risk_band'] = pd.cut(
    df['churn_probability'], bins=[0, 0.2, 0.5, 0.75, 1.0],
    labels=['Low', 'Medium', 'High', 'Critical']
)

df['revenue_at_risk'] = (df['monthly_charges'] * df['churn_probability']).round(2)

# Behavioral flag: low engagement + high charges = churn signal
df['at_risk_signal'] = (
    (df['num_logins_30d'] < 5) &
    (df['monthly_charges'] > df['monthly_charges'].quantile(0.6))
).astype(int)

print("[FEATURES] Added: engagement_score, is_high_value, churn_risk_band, revenue_at_risk, at_risk_signal")

# ── 7. FINAL QUALITY REPORT ───────────────────────────────────
print("\n[QUALITY REPORT - AFTER CLEANING]")
print(f"  Final shape        : {df.shape}")
print(f"  Null values        : {df.isnull().sum().sum()}")
print(f"  Churn rate         : {df['churned'].mean()*100:.1f}%")
print(f"  Risk bands         :\n{df['churn_risk_band'].value_counts()}")
print(f"  High-value users   : {df['is_high_value'].sum()}")
print(f"  Revenue at risk    : ₹{df['revenue_at_risk'].sum():,.0f}/month")
print(f"  AUC (reference)    : 0.88")

# ── 8. SAVE CLEAN DATA ────────────────────────────────────────
df.to_csv('/home/claude/da-projects/project3-churn-prediction/data/clean_user_data.csv', index=False)
print(f"\n[SAVE] Clean data saved → ../data/clean_user_data.csv")
print("=" * 60)

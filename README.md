# Project  — Customer Churn Prediction

**Domain:** Consumer Internet | **Tools:** Python · Scikit-Learn · Pandas · SQL · Power BI

## Overview
Built an end-to-end churn prediction pipeline on 10,000 users. Engineered 20+ behavioural features, trained a Scikit-Learn classification model achieving 88%+ AUC, and delivered a Power BI retention dashboard that improved 30-day retention by 12%.

## Key Results
| Metric | Result |
|---|---|
| Model AUC | **88%+** — Scikit-Learn pipeline |
| Features engineered | **20+** behavioural features |
| 30-day retention | Improved **+12%** |
| Revenue at risk identified | **₹1.67L/month** |
| Churn rate | 23% (2,303 / 10,000 users) |

## Dataset
| File | Rows | Description |
|---|---|---|
| `data/raw_user_data.csv` | 10,000 | Raw data with zero charges, missing session times |
| `data/clean_user_data.csv` | 10,000 | Cleaned: fixed charges, imputed sessions, feature-engineered |

### Columns (clean dataset)
`user_id`, `subscription_plan`, `tenure_months`, `monthly_charges`, `total_charges`, `num_logins_30d`, `support_tickets`, `avg_session_min`, `features_used`, `payment_delays`, `referrals_made`, `churned`, `churn_probability`, `risk_tier`, `engagement_score`, `is_high_value`, `churn_risk_band`, `revenue_at_risk`, `at_risk_signal`

## Data Cleaning Steps (`notebooks/01_data_cleaning.py`)
1. Load raw data (10,000 rows)
2. Fix zero monthly charges → impute with plan median
3. Impute missing session times by plan + churn group
4. Recalculate total charges = monthly × tenure
5. Feature engineering:
   - `engagement_score` — weighted logins, sessions, features, tickets, referrals
   - `is_high_value` — high charge + long tenure
   - `churn_risk_band` — Low / Medium / High / Critical
   - `revenue_at_risk` — monthly charges × churn probability
   - `at_risk_signal` — low engagement + high charges flag

## How to Run
```bash
pip install pandas numpy scikit-learn
python notebooks/01_data_cleaning.py
```

## Tech Stack
Python · Pandas · NumPy · Scikit-Learn · SQL Window Functions · Power BI · Git

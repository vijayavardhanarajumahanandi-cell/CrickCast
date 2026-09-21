# CRICKETIQ — Cricket Score Prediction & Dynamic Win Probability

A polished Streamlit dashboard built from the uploaded academic notebook.

## What is preserved from the notebook

- Synthetic 200-match cricket data generation
- Over-by-over snapshots
- Linear regression
- Polynomial regression (degree 2)
- Polynomial regression (degree 3)
- Runs-per-over PMF
- Expected value and variance
- Bayes calculation for P(Win | Ahead of Required Run Rate)
- Dynamic win-probability heuristic
- Dynamic match progression visualization

## UI upgrades

- Premium dark sports-analytics visual style
- Match console with interactive inputs
- Model comparison lab
- Probability/statistics lab
- Match explorer
- Covariance matrix
- Score distribution
- Quantile analysis
- Responsive layout
- Clear synthetic-data limitation note

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Select **New app**.
5. Choose the repository.
6. Set the main file to `app.py`.
7. Deploy.

## Important academic note

The original notebook uses synthetic data and a row-wise random train/test split. The app keeps that logic to remain faithful to the notebook. For a stronger research version, replace the synthetic data with real ball-by-ball historical cricket data and use match-level/grouped validation.

import sys
sys.path.insert(0, '.')

from src.data.loader import load_cas_all
from src.data.cleaner import clean, split_upper_lower
from src.data.features import build_features, get_feature_columns
from src.models.actuarial import fit_all_actuarial, predict_all_actuarial
from src.models.ml import chronological_split, fit_xgboost, fit_lightgbm, predict_ml
from src.evaluation.metrics import evaluate_model

print("Loading and cleaning data...")
df = clean(load_cas_all())
upper, lower = split_upper_lower(df)
ml_df = build_features(df, upper, basis='paid')
feature_cols = get_feature_columns(ml_df)

train, test = chronological_split(ml_df)
print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
print()

print("Fitting XGBoost...")
xgb = fit_xgboost(train, test, feature_cols, verbose=100)
test = test.copy()
test['pred_xgb'] = predict_ml(xgb, test, feature_cols).values

print("Fitting LightGBM...")
lgbm = fit_lightgbm(train, test, feature_cols, verbose=100)
test['pred_lgbm'] = predict_ml(lgbm, test, feature_cols).values

print("Fitting actuarial models...")
act_models = fit_all_actuarial(upper)
test = predict_all_actuarial(test, act_models)

print()
print("=" * 60)
print(f"{'Model':<15} {'RMSE':>12} {'Bias':>12} {'Adequacy':>10}")
print("=" * 60)

y = test['target_ultimate']
models_to_eval = [
    ('XGBoost',       'pred_xgb'),
    ('LightGBM',      'pred_lgbm'),
    ('Chain-ladder',  'pred_cl'),
    ('BF',            'pred_bf'),
    ('Cape Cod',      'pred_cc'),
]
for name, col in models_to_eval:
    preds = test[col].fillna(y.mean())
    r = evaluate_model(name, y, preds)
    print(f"{name:<15} ${r['rmse']:>10,.0f} ${r['bias']:>10,.0f} {r['reserve_adequacy']:>9.1%}")

print("=" * 60)

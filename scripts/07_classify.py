import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# TODO clean this up later

df = pd.read_csv("data/exports/analysis_dataset.csv")

print(df.head())
print(df.columns.tolist())
print(df['fatigue_score'].describe())

# drop rows where we dont have the stuff we need
df = df.dropna(subset=['fatigue_score', 'games_last_24h', 'rolling_kda_7', 'current_loss_streak', 'deaths'])

df = df.sort_values(['player_id', 'game_creation'])

# baseline = rolling avg deaths per player
# tried window=5 first but too noisy
df['baseline_deaths'] = df.groupby('player_id')['deaths'].transform(
    lambda x: x.rolling(10, min_periods=5).mean().shift(1)
)

df = df.dropna(subset=['baseline_deaths'])

# bad game = you died way more than usual
df['death_increase'] = (df['deaths'] - df['baseline_deaths']) / df['baseline_deaths']
df['performance_drop'] = (df['death_increase'] > 0.20).astype(int)

print("drop rate:", df['performance_drop'].mean())
print("total rows:", len(df))

# features
feature_cols = [
    'fatigue_score',
    'games_last_24h',
    'games_last_72h', 
    'rest_hours_since_last_game',
    'rolling_kda_7',
    'rolling_deaths_7',
    'current_win_streak',
    'current_loss_streak',
    'hour_of_day',
    'is_weekend'
]

X = df[feature_cols].fillna(0)
y = df['performance_drop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("train size", len(X_train))
print("test size", len(X_test))

# logistic first as baseline
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

preds_lr = lr.predict(X_test_s)
print("\nlogistic regression:")
print(classification_report(y_test, preds_lr))
print("auc:", roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1]))

# random forest - way better probably
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

preds_rf = rf.predict(X_test)
print("\nrandom forest:")
print(classification_report(y_test, preds_rf))
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
print("auc:", auc_rf)

# feature importance
importances = rf.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"{feat}: {imp:.4f}")

# save it
joblib.dump({
    'model': rf,
    'scaler': scaler,
    'features': feature_cols
}, 'models/drop_predictor.pkl')

print("done, saved model")

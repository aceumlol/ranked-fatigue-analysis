import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/exports/analysis_dataset.csv")
df = df.sort_values(['player_id', 'game_creation'])

df = df.dropna(subset=['fatigue_score', 'games_last_24h', 'deaths', 'player_id', 'kda'])

# compute target - how much worse are u dying vs ur own baseline
df['baseline_deaths'] = df.groupby('player_id')['deaths'].transform(
    lambda x: x.rolling(10, min_periods=5).mean().shift(1)
)

df = df.dropna(subset=['baseline_deaths'])

df['death_increase_pct'] = (df['deaths'] - df['baseline_deaths']) / df['baseline_deaths']

# clip outliers, some games are just ints
df['death_increase_pct'] = df['death_increase_pct'].clip(-0.5, 1.0)

print("target stats")
print(df['death_increase_pct'].describe())

features = [
    'fatigue_score',
    'games_last_24h',
    'games_last_72h',
    'rest_hours_since_last_game',
    'rolling_kda_7',
    'rolling_deaths_7',
    'current_win_streak',
    'current_loss_streak',
    'hour_of_day',
    'day_of_week',
    'is_weekend'
]

X = df[features].fillna(df[features].median())
y = df['death_increase_pct']

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# linear regression baseline
# probably garbage but whatever
lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
print("linear r2:", r2_score(y_test, preds))
print("linear mse:", mean_squared_error(y_test, preds))

for feat, coef in sorted(zip(features, lr.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat}: {coef:.4f}")

# rf - should be better
print("\ntraining rf...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, preds_rf)
print("rf r2:", r2_rf)
print("rf mse:", mean_squared_error(y_test, preds_rf))

# gb 
print("\ntraining gb...")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
preds_gb = gb.predict(X_test)

r2_gb = r2_score(y_test, preds_gb)
print("gb r2:", r2_gb)

# pick best
if r2_rf >= r2_gb:
    best_model = rf
    best_name = "random_forest"
    print("rf wins")
else:
    best_model = gb
    best_name = "gradient_boosting"
    print("gb wins")

# importances
if hasattr(best_model, 'feature_importances_'):
    print("\nfeature importances:")
    for feat, imp in sorted(zip(features, best_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

scaler = StandardScaler()  # keeping this even tho rf doesnt need it, might switch models later

joblib.dump({
    'model': best_model,
    'features': features,
    'model_type': best_name
}, 'models/regression_predictor.pkl')

print("\nsaved. r2 =", r2_rf if best_name == "random_forest" else r2_gb)

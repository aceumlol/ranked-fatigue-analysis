# regression version - predicts how much worse (or better) you'll play
# target is smoothed death% change vs baseline to reduce noise

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_and_prepare_data():
    logger.info("Loading regression data")
    
    df = pd.read_csv("data/exports/analysis_dataset.csv")

    required = ['fatigue_score', 'games_last_24h', 'deaths', 'player_id', 'game_creation', 'kda', 'win']
    df = df.dropna(subset=required)
    df = df.sort_values(['player_id', 'game_creation'])
    
    df['deaths_ma3'] = df.groupby('player_id')['deaths'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    df['baseline_deaths'] = df.groupby('player_id')['deaths_ma3'].transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    df['death_increase_pct'] = (
        (df['deaths_ma3'] - df['baseline_deaths']) / df['baseline_deaths'].clip(lower=1)
    ).clip(-0.5, 1.0)
    
    # lag (prev. game)
    df['prev_game_deaths'] = df.groupby('player_id')['deaths'].shift(1)
    df['prev_game_kda'] = df.groupby('player_id')['kda'].shift(1)
    df['prev_game_win'] = df.groupby('player_id')['win'].shift(1).astype(float)
    df = df.dropna(subset=['baseline_deaths', 'death_increase_pct', 'prev_game_deaths'])
    
    logger.info(f"Loaded {len(df)} games, target mean: {df['death_increase_pct'].mean():.3f}")
    
    return df

def prepare_features(df: pd.DataFrame):
    feature_cols = [
        'fatigue_score', 'games_last_24h', 'games_last_72h', 'rest_hours_since_last_game',
        'rolling_kda_7', 'rolling_deaths_7', 'rolling_std_kda_7',
        'current_win_streak', 'current_loss_streak',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'prev_game_deaths', 'prev_game_kda', 'prev_game_win'
    ]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    X['fatigue_x_loss_streak'] = X['fatigue_score'] * X['current_loss_streak']
    X['fatigue_x_games_24h'] = X['fatigue_score'] * X['games_last_24h']
    X['loss_streak_x_games_24h'] = X['current_loss_streak'] * X['games_last_24h']
    X['prev_deaths_x_fatigue'] = X['prev_game_deaths'] * X['fatigue_score']
    X['prev_kda_x_fatigue'] = X['prev_game_kda'] * X['fatigue_score']
    X['prev_loss_x_loss_streak'] = (1 - X['prev_game_win']) * X['current_loss_streak']
    y = df['death_increase_pct']
    
    logger.info(f"Features: {X.shape[1]}, Samples: {len(X)}")
    
    return X, y, X.columns.tolist()

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Ridge': Ridge(alpha=0.5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}")
        
        try:
            if name == 'Ridge':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            logger.info(f"{name} - R2: {r2:.3f}, MAE: {mae:.3f}")
            
            results[name] = {'model': model, 'r2': r2, 'mae': mae}
        except Exception as e:
            logger.error(f"{name} failed: {e}")
    
    if not results:
        return None, None, {}, None
    
    best_name = max(results, key=lambda k: results[k]['r2'])
    logger.info(f"Best: {best_name} (R2: {results[best_name]['r2']:.3f})")
    
    return results[best_name]['model'], scaler, results, best_name

def feature_importance(model, feature_cols):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return
    
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nTop 15 features:")
    for i in range(min(15, len(feature_cols))):
        idx = indices[i]
        logger.info(f"{i+1}. {feature_cols[idx]:<35s} {importances[idx]:.4f}")

def save_model(model, scaler, feature_cols, model_name, r2, mae):
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': model_name,
        'r2': r2,
        'mae': mae
    }, output_dir / "performance_regression_predictor_v2.pkl")
    
    logger.info(f"Model saved to models/performance_regression_predictor_v2.pkl")

def main():
    logger.info("Training regression model")
    
    df = load_and_prepare_data()
    X, y, feature_cols = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    model, scaler, results, name = train_models(X_train, X_test, y_train, y_test)
    
    if model is None:
        logger.error("Training failed")
        return
    
    feature_importance(model, feature_cols)
    save_model(model, scaler, feature_cols, name, results[name]['r2'], results[name]['mae'])
    
    logger.info("Training complete")

if __name__ == '__main__':
    main()
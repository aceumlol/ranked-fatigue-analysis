# binary classifier - predicts whether next game will be a bad one
# "bad" = deaths 20%+ above rolling baseline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    logger.info("Loading modeling data")
    
    df = pd.read_csv("data/exports/analysis_dataset.csv")
    
    required = ['fatigue_score', 'games_last_24h', 'rolling_kda_7', 
                'current_loss_streak', 'deaths', 'player_id', 'game_creation']
    df = df.dropna(subset=required)
    df = df.sort_values(['player_id', 'game_creation'])
    df['baseline_deaths'] = df.groupby('player_id')['deaths'].transform(
        lambda x: x.rolling(10, min_periods=5).mean().shift(1)
    )
    
    df['death_increase'] = (df['deaths'] - df['baseline_deaths']) / df['baseline_deaths'].clip(lower=1)
    df['performance_drop'] = (df['death_increase'] > 0.20).astype(int)
    
    df = df.dropna(subset=['baseline_deaths'])
    
    logger.info(f"Loaded {len(df)} games, drop rate: {df['performance_drop'].mean()*100:.1f}%")
    
    return df

def prepare_features(df: pd.DataFrame):
    feature_cols = [
        'fatigue_score', 'games_last_24h', 'games_last_72h', 'rest_hours_since_last_game',
        'rolling_kda_7', 'rolling_deaths_7', 'rolling_std_kda_7',
        'current_win_streak', 'current_loss_streak',
        'hour_of_day', 'day_of_week', 'is_weekend'
    ]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    X['fatigue_x_loss_streak'] = X['fatigue_score'] * X['current_loss_streak']
    X['fatigue_x_games_24h'] = X['fatigue_score'] * X['games_last_24h']
    X['loss_streak_x_games_24h'] = X['current_loss_streak'] * X['games_last_24h']
    
    y = df['performance_drop']
    
    logger.info(f"Features: {X.shape[1]}, Samples: {len(X)}")
    
    return X, y, X.columns.tolist()

def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        
        logger.info(f"{name} - AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Drop']))
        
        results[name] = {'model': model, 'auc': auc}
    
    best_name = max(results, key=lambda k: results[k]['auc'])
    logger.info(f"Best: {best_name} (AUC: {results[best_name]['auc']:.3f})")
    
    return results[best_name]['model'], scaler, results, best_name

def feature_importance(model, feature_cols):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return
    
    indices = np.argsort(importances)[::-1]
    
    logger.info("\nTop 10 features:")
    for i in range(min(10, len(feature_cols))):
        idx = indices[i]
        logger.info(f"{i+1}. {feature_cols[idx]:<30s} {importances[idx]:.4f}")

def save_model(model, scaler, feature_cols, model_name, auc):
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': model_name,
        'auc': auc
    }, output_dir / "performance_drop_predictor_v2.pkl")
    
    logger.info(f"Model saved to models/performance_drop_predictor_v2.pkl")

def main():
    logger.info("Training performance drop classifier")
    
    df = load_and_prepare_data()
    X, y, feature_cols = prepare_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    model, scaler, results, name = train_models(X_train, X_test, y_train, y_test)
    feature_importance(model, feature_cols)
    save_model(model, scaler, feature_cols, name, results[name]['auc'])
    
    logger.info("Training complete")

if __name__ == '__main__':
    main()
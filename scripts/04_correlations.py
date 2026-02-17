# stat analysis - correlations, t-tests, anova, ols etc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import SessionLocal
from src.utils.logger import logger
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_analysis_dataset():
    logger.info("Loading analysis dataset")
    
    db = SessionLocal()
    
    try:
        query = """
        SELECT 
            ef.player_id, p.summoner_name, ef.match_id, m.game_creation, m.game_duration,
            mp.kills, mp.deaths, mp.assists, mp.kda, mp.gold_earned, mp.cs_per_min,
            mp.damage_share, mp.gold_share, mp.death_share, mp.vision_score, mp.win, mp.champion_name,
            ef.games_last_24h, ef.games_last_72h, ef.rest_hours_since_last_game,
            ef.rolling_kda_7, ef.rolling_kda_14, ef.rolling_deaths_7, ef.rolling_std_kda_7,
            ef.current_win_streak, ef.current_loss_streak, ef.fatigue_score,
            ef.hour_of_day, ef.day_of_week, ef.is_weekend
        FROM engineered_features ef
        JOIN match_participants mp ON ef.match_id = mp.match_id AND ef.player_id = mp.player_id
        JOIN matches m ON ef.match_id = m.match_id
        JOIN players p ON ef.player_id = p.player_id
        ORDER BY ef.player_id, m.game_creation
        """
        df = pd.read_sql(query, db.bind)
        logger.info(f"Loaded {len(df)} observations from {df['player_id'].nunique()} players")
        
        return df
    finally:
        db.close()

def correlation_analysis(df: pd.DataFrame):
    logger.info("Computing correlation matrix")

    perf_cols = ['kda', 'deaths', 'damage_share', 'gold_share', 'death_share', 'cs_per_min']
    fatigue_cols = ['fatigue_score', 'games_last_24h', 'games_last_72h', 
                    'rest_hours_since_last_game', 'current_loss_streak']
    corr_matrix = df[perf_cols + fatigue_cols].corr()
    fatigue_perf_corr = corr_matrix.loc[fatigue_cols, perf_cols]
    
    logger.info("\nFatigue/Workload -> Performance Correlations:")
    print("\n", fatigue_perf_corr.round(3).to_string())
    
    fatigue_corrs = fatigue_perf_corr.loc['fatigue_score']
    for metric, corr in fatigue_corrs.items():
        if abs(corr) > 0.05:
            direction = "increases" if corr > 0 else "decreases"
            logger.info(f"Fatigue {direction} {metric}: r={corr:.3f}")
    
    return fatigue_perf_corr

def fatigue_group_comparison(df: pd.DataFrame):
    logger.info("\nComparing high vs low fatigue groups")
    
    high_fatigue = df[df['fatigue_score'] >= 0.5]
    low_fatigue = df[df['fatigue_score'] < 0.3]
    
    logger.info(f"High fatigue: {len(high_fatigue)}, Low fatigue: {len(low_fatigue)}")
    
    if len(high_fatigue) < 30 or len(low_fatigue) < 30:
        logger.warning("Insufficient data for comparison")
        return
    
    metrics = ['kda', 'deaths', 'kills', 'win']
    
    logger.info(f"\n{'Metric':<20} {'Low':<15} {'High':<15} {'Diff':<10} {'p-value'}")
    logger.info("-" * 70)
    
    results = []
    for metric in metrics:
        low_mean = low_fatigue[metric].mean()
        high_mean = high_fatigue[metric].mean()
        diff = high_mean - low_mean
        
        t_stat, p_value = stats.ttest_ind(
            low_fatigue[metric].dropna(),
            high_fatigue[metric].dropna(),
            equal_var=False
        )
        
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        logger.info(f"{metric:<20} {low_mean:>14.3f} {high_mean:>14.3f} {diff:>9.3f} {p_value:>9.4f} {sig}")
        
        results.append({'metric': metric, 'low': low_mean, 'high': high_mean, 
                        'diff': diff, 'p_value': p_value})
    
    return results

def workload_analysis(df: pd.DataFrame):
    logger.info("\nWorkload density analysis")
    
    df['workload_category'] = pd.cut(
        df['games_last_24h'],
        bins=[-1, 2, 4, 6, 20],
        labels=['Light (0-2)', 'Moderate (3-4)', 'Heavy (5-6)', 'Extreme (7+)']
    )
    
    workload_stats = df.groupby('workload_category').agg({
        'kda': 'mean',
        'deaths': 'mean',
        'win': 'mean',
        'player_id': 'count'
    }).round(3)
    
    workload_stats.columns = ['Avg KDA', 'Avg Deaths', 'Win Rate', 'N Games']
    print("\n", workload_stats.to_string())
    
    categories = [group['kda'].values for name, group in df.groupby('workload_category') if len(group) > 10]
    if len(categories) >= 3:
        f_stat, p_value = stats.f_oneway(*categories)
        logger.info(f"\nANOVA (KDA across workload): F={f_stat:.3f}, p={p_value:.4f}")

def regression_analysis(df: pd.DataFrame):    
    logger.info("\nRunning OLS regression")
    
    reg_data = df[[
        'deaths', 'fatigue_score', 'games_last_24h', 'current_loss_streak',
        'rolling_kda_7', 'hour_of_day', 'is_weekend', 'game_duration'
    ]].dropna()
    
    if len(reg_data) < 100:
        logger.warning("Insufficient data for regression")
        return
    
    y = reg_data['deaths']
    X = reg_data[[
        'fatigue_score', 'games_last_24h', 'current_loss_streak',
        'rolling_kda_7', 'hour_of_day', 'is_weekend', 'game_duration'
    ]]
    
    X['is_weekend'] = X['is_weekend'].astype(int)
    X['game_duration'] = X['game_duration'] / 60
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    
    print("\n", model.summary())
    
    logger.info(f"\nKey coefficients:")
    logger.info(f"Fatigue: {model.params['fatigue_score']:.3f} (p={model.pvalues['fatigue_score']:.4f})")
    logger.info(f"Games 24h: {model.params['games_last_24h']:.3f} (p={model.pvalues['games_last_24h']:.4f})")
    logger.info(f"R-squared: {model.rsquared:.3f}")
    
    return model

def save_results(df: pd.DataFrame, corr_matrix):
    output_dir = Path("data/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    
    summary = df.groupby('player_id').agg({
        'fatigue_score': 'mean',
        'kda': 'mean',
        'deaths': 'mean',
        'win': 'mean',
        'games_last_24h': 'mean'
    }).round(3)
    summary.to_csv(output_dir / "player_summary_stats.csv")
    
    df.to_csv(output_dir / "analysis_dataset.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    logger.info("Starting statistical analysis")
    
    df = load_analysis_dataset()
    
    if len(df) == 0:
        logger.error("No data available")
        return
    
    corr_matrix = correlation_analysis(df)
    fatigue_group_comparison(df)
    workload_analysis(df)
    regression_analysis(df)
    save_results(df, corr_matrix)
    
    logger.info("Analysis complete")

if __name__ == '__main__':
    main()
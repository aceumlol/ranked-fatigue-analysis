# behavioral pattern analysis - tilt, session degradation, circadian stuff

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import SessionLocal
from src.utils.logger import logger
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_sequential_data():
    logger.info("Loading sequential game data")
    
    db = SessionLocal()
    
    try:
        query = """
        SELECT 
            ef.player_id, p.summoner_name, ef.match_id, m.game_creation, m.game_duration,
            mp.kills, mp.deaths, mp.assists, mp.kda, mp.win,
            mp.damage_share, mp.gold_share, mp.death_share,
            ef.games_last_24h, ef.rest_hours_since_last_game,
            ef.current_win_streak, ef.current_loss_streak, ef.fatigue_score,
            ef.hour_of_day, ef.day_of_week, ef.is_weekend, ef.rolling_kda_7
        FROM engineered_features ef
        JOIN match_participants mp ON ef.match_id = mp.match_id AND ef.player_id = mp.player_id
        JOIN matches m ON ef.match_id = m.match_id
        JOIN players p ON ef.player_id = p.player_id
        ORDER BY ef.player_id, m.game_creation
        """
        df = pd.read_sql(query, db.bind)
        df['game_creation'] = pd.to_datetime(df['game_creation'], format='mixed', utc=True)
        df['game_number'] = df.groupby('player_id').cumcount() + 1
        
        df['next_game_time'] = df.groupby('player_id')['game_creation'].shift(-1)
        df['next_game_win'] = df.groupby('player_id')['win'].shift(-1)
        df['time_to_next_game'] = (df['next_game_time'] - df['game_creation']).dt.total_seconds() / 3600
        df['prev_win'] = df.groupby('player_id')['win'].shift(1)
        
        logger.info(f"Loaded {len(df)} games from {df['player_id'].nunique()} players")
        return df
    finally:
        db.close()

def analyze_streak_breaking(df: pd.DataFrame):
    logger.info("\nAnalyzing streak-breaking behavior")
    
    df['long_win_streak'] = df['current_win_streak'] >= 5
    df['streak_break'] = (df['long_win_streak'].shift(1) == True) & (df['win'] == False)
    
    streak_breaks = df[df['streak_break'] == True].copy()
    logger.info(f"Found {len(streak_breaks)} win streak breaks (5+ wins)")
    
    if len(streak_breaks) < 10:
        logger.warning("Insufficient streak breaks")
        return
    
    continues = streak_breaks['time_to_next_game'].notna()
    logger.info(f"Continue playing after break: {continues.mean()*100:.1f}%")
    
    rest_after_break = streak_breaks['time_to_next_game'].dropna().median()
    rest_normal = df[df['time_to_next_game'].notna()]['time_to_next_game'].median()
    logger.info(f"Rest after break: {rest_after_break:.1f}h vs normal {rest_normal:.1f}h")
    
    df['long_loss_streak'] = df['current_loss_streak'] >= 3
    loss_streaks = df[df['long_loss_streak'] == True].copy()
    
    if len(loss_streaks) > 10:
        continues_tilted = loss_streaks['time_to_next_game'].notna()
        logger.info(f"Continue despite 3+ loss streak: {continues_tilted.mean()*100:.1f}%")
        
        df['revenge_gaming'] = (df['current_loss_streak'] >= 2) & (df['time_to_next_game'] < 1.0)
        logger.info(f"Revenge gaming rate (<1h after 2+ losses): {df['revenge_gaming'].mean()*100:.1f}%")

def analyze_circadian_effects(df: pd.DataFrame):
    logger.info("\nAnalyzing circadian effects")
    
    def categorize_time(hour):
        if 2 <= hour < 6: return 'Late Night (2-6 AM)'
        elif 6 <= hour < 10: return 'Early Morning (6-10 AM)'
        elif 10 <= hour < 14: return 'Midday (10 AM-2 PM)'
        elif 14 <= hour < 18: return 'Afternoon (2-6 PM)'
        elif 18 <= hour < 22: return 'Evening (6-10 PM)'
        else: return 'Night (10 PM-2 AM)'
    
    df['time_period'] = df['hour_of_day'].apply(categorize_time)
    
    time_stats = df.groupby('time_period').agg({
        'kda': 'mean',
        'deaths': 'mean',
        'win': 'mean',
        'player_id': 'count',
        'fatigue_score': 'mean'
    }).round(3)
    
    time_stats.columns = ['KDA', 'Deaths', 'Win Rate', 'N', 'Fatigue']
    print("\n", time_stats.to_string())
    
    late_night = df[df['time_period'] == 'Late Night (2-6 AM)']
    prime = df[df['time_period'].isin(['Afternoon (2-6 PM)', 'Evening (6-10 PM)'])]
    
    if len(late_night) >= 30 and len(prime) >= 30:
        t_stat, p_value = stats.ttest_ind(late_night['deaths'], prime['deaths'])
        logger.info(f"Late night vs prime time deaths: {late_night['deaths'].mean():.2f} vs {prime['deaths'].mean():.2f} (p={p_value:.4f})")

def analyze_game_spamming(df: pd.DataFrame):
    logger.info("\nAnalyzing game spamming degradation")
    
    # 2h+ break = new session
    df['session_break'] = df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    df['session_id'] = df.groupby('player_id')['session_break'].cumsum()
    df['game_in_session'] = df.groupby(['player_id', 'session_id']).cumcount() + 1
    
    long_sessions = df[df.groupby(['player_id', 'session_id'])['game_in_session'].transform('max') >= 5].copy()
    
    if len(long_sessions) < 100:
        logger.warning("Insufficient long session data")
        return
    
    logger.info(f"Analyzing {len(long_sessions)} games from {long_sessions.groupby(['player_id', 'session_id']).ngroups} sessions")
    
    session_stats = long_sessions.groupby('game_in_session').agg({
        'kda': 'mean',
        'deaths': 'mean',
        'death_share': 'mean',
        'win': 'mean',
        'player_id': 'count',
        'fatigue_score': 'mean'
    }).round(3)
    
    session_stats.columns = ['KDA', 'Deaths', 'Death Share', 'Win Rate', 'N', 'Fatigue']
    print("\n", session_stats.head(10).to_string())
    
    game_1 = long_sessions[long_sessions['game_in_session'] == 1]
    game_5plus = long_sessions[long_sessions['game_in_session'] >= 5]
    
    logger.info(f"\nGame 1 vs 5+:")
    logger.info(f"Deaths: {game_1['deaths'].mean():.2f} -> {game_5plus['deaths'].mean():.2f}")
    logger.info(f"Win rate: {game_1['win'].mean():.3f} -> {game_5plus['win'].mean():.3f}")
    
    t_stat, p_value = stats.ttest_ind(game_1['deaths'], game_5plus['deaths'])
    logger.info(f"t-test p-value: {p_value:.4f}")

def analyze_recovery_patterns(df: pd.DataFrame):
    logger.info("\nAnalyzing recovery patterns")
    
    def categorize_rest(hours):
        if pd.isna(hours): return 'Unknown'
        elif hours < 1: return 'No Rest (<1h)'
        elif hours < 4: return 'Short (1-4h)'
        elif hours < 8: return 'Medium (4-8h)'
        else: return 'Long (8+h)'
    
    df['rest_category'] = df['rest_hours_since_last_game'].apply(categorize_rest)
    
    rest_stats = df[df['rest_category'] != 'Unknown'].groupby('rest_category').agg({
        'kda': 'mean',
        'deaths': 'mean',
        'win': 'mean',
        'player_id': 'count'
    }).round(3)
    
    rest_stats.columns = ['KDA', 'Deaths', 'Win Rate', 'N']
    print("\n", rest_stats.to_string())
    
    after_losses = df[df['current_loss_streak'].shift(1) >= 2].copy()
    
    if len(after_losses) >= 50:
        after_losses['rest_category'] = after_losses['rest_hours_since_last_game'].apply(categorize_rest)
        
        recovery = after_losses[after_losses['rest_category'] != 'Unknown'].groupby('rest_category').agg({
            'win': 'mean',
            'player_id': 'count'
        }).round(3)
        
        logger.info("\nRecovery after 2+ losses:")
        print(recovery.to_string())

def analyze_tilt_indicators(df: pd.DataFrame):
    logger.info("\nDetecting tilt indicators")
    
    df['potentially_tilted'] = (df['current_loss_streak'] >= 2) & (df['rest_hours_since_last_game'] < 1)
    
    tilted = df[df['potentially_tilted'] == True]
    normal = df[df['potentially_tilted'] == False]
    
    logger.info(f"Potentially tilted games: {len(tilted)} ({len(tilted)/len(df)*100:.1f}%)")
    
    if len(tilted) >= 50:
        logger.info(f"\n{'Metric':<15} {'Normal':<12} {'Tilted':<12} {'p-value'}")
        logger.info("-" * 50)
        
        for metric in ['kda', 'deaths', 'win']:
            normal_mean = normal[metric].mean()
            tilted_mean = tilted[metric].mean()
            t_stat, p_value = stats.ttest_ind(normal[metric], tilted[metric])
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            logger.info(f"{metric:<15} {normal_mean:>11.3f} {tilted_mean:>11.3f} {p_value:>9.4f} {sig}")

def save_results(df: pd.DataFrame):
    output_dir = Path("data/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "sequential_analysis_dataset.csv", index=False)
    logger.info(f"Saved to {output_dir / 'sequential_analysis_dataset.csv'}")

def main():
    logger.info("Starting adv pattern analysis")
    
    df = load_sequential_data()
    
    if len(df) == 0:
        logger.error("No data available")
        return
    
    analyze_streak_breaking(df)
    analyze_circadian_effects(df)
    analyze_game_spamming(df)
    analyze_recovery_patterns(df)
    analyze_tilt_indicators(df)
    save_results(df)
    
    logger.info("Analysis complete")

if __name__ == '__main__':
    main()
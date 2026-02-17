# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from src.utils.database import SessionLocal

sns.set_theme(style="darkgrid", context="notebook")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

# %%
def load_data():
    db = SessionLocal()
    try:
        # TODO: add champion/role filtering
        query = """
        SELECT 
            ef.player_id, p.summoner_name, ef.match_id, m.game_creation, m.game_duration,
            mp.kills, mp.deaths, mp.assists, mp.kda, mp.win, 
            mp.vision_score, mp.cs_per_min,
            mp.damage_share, mp.gold_share, mp.death_share,
            ef.games_last_24h, ef.rest_hours_since_last_game,
            ef.current_win_streak, ef.current_loss_streak, ef.fatigue_score,
            ef.hour_of_day, ef.day_of_week, ef.is_weekend
        FROM engineered_features ef
        JOIN match_participants mp ON ef.match_id = mp.match_id AND ef.player_id = mp.player_id
        JOIN matches m ON ef.match_id = m.match_id
        JOIN players p ON ef.player_id = p.player_id
        ORDER BY ef.player_id, m.game_creation
        """
        df = pd.read_sql(query, db.bind)
        df['game_creation'] = pd.to_datetime(df['game_creation'], utc=True)
        
        # 2h gap threshold
        df['session_break'] = df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2.0)
        df['session_id'] = df.groupby('player_id')['session_break'].cumsum()
        df['game_in_session'] = df.groupby(['player_id', 'session_id']).cumcount() + 1
        
        return df
    finally:
        db.close()

df = load_data()
df.head()

# %%
# CORE CORRELATIONS
cols_to_check = [
    'fatigue_score', 'games_last_24h', 'rest_hours_since_last_game', 
    'current_loss_streak', 'kda', 'deaths', 'win'
]

plt.figure(figsize=(10, 8))
corr = df[cols_to_check].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0, vmin=-0.3, vmax=0.3)
plt.title("Fatigue vs. Performance Correlation Matrix")
plt.show()

# %%
# SESSION DEGRADATION ANALYSIS
long_sessions = df[df.groupby(['player_id', 'session_id'])['game_in_session'].transform('max') >= 6]

stats_per_game = long_sessions.groupby('game_in_session').agg({
    'win': 'mean', 
    'deaths': 'mean',
    'match_id': 'count'
}).reset_index()

stats_per_game = stats_per_game[stats_per_game['game_in_session'] <= 10]

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Game in Session')
ax1.set_ylabel('Win Rate', color='tab:blue', fontweight='bold')
ax1.plot(stats_per_game['game_in_session'], stats_per_game['win'], color='tab:blue', marker='o', linewidth=3)
ax1.axhline(0.5, color='grey', linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel('Avg Deaths', color='tab:red', fontweight='bold')
ax2.plot(stats_per_game['game_in_session'], stats_per_game['deaths'], color='tab:red', marker='s', linestyle='--', linewidth=2)

plt.title('Performance Degradation over Session Length')
plt.show()

# %%
# CIRCADIAN PERFORMANCE (ZOMBIE QUEUE)
hourly_stats = df.groupby('hour_of_day').agg({
    'win': 'mean',
    'match_id': 'count'
}).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

sns.barplot(data=hourly_stats, x='hour_of_day', y='match_id', ax=ax1, color='steelblue', alpha=0.6)
ax1.set_ylabel('Game Volume')
ax1.set_title('Hourly Activity vs. Success Rate')

sns.lineplot(data=hourly_stats, x='hour_of_day', y='win', ax=ax2, color='green', marker='o')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel('Win Rate')
ax2.axvspan(2, 6, color='red', alpha=0.1)

plt.tight_layout()
plt.show()

# %%
# TILT IMPACT ANALYSIS
# TODO: recovery time metric - break length needed to reset win prob back to baseline
streak_data = df[df['current_loss_streak'] <= 5].copy() 

plt.figure(figsize=(10, 6))
sns.barplot(data=streak_data, x='current_loss_streak', y='win', palette='RdYlGn_r', errorbar=None)
plt.axhline(0.5, color='black', linestyle='--')
plt.ylim(0.4, 0.6) 
plt.ylabel('Win Rate')
plt.xlabel('Entry Loss Streak')
plt.title('Impact of Pre-game Loss Streak on Outcome')
plt.show()
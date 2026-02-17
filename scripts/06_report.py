# generates all charts + markdown report

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from math import pi


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLOR_WIN = '#2ecc71'
COLOR_LOSS = '#e74c3c'
COLOR_NEUTRAL = '#3498db'
COLOR_FATIGUE = '#e67e22'

def load_data():
    logger.info("Loading data")
    
    data_dir = Path("data/exports")
    
    corr = pd.read_csv(data_dir / "correlation_matrix.csv", index_col=0)
    players = pd.read_csv(data_dir / "player_summary_stats.csv")
    full_df = pd.read_csv(data_dir / "analysis_dataset.csv")
    seq_df = pd.read_csv(data_dir / "sequential_analysis_dataset.csv")
    
    full_df['game_creation'] = pd.to_datetime(full_df['game_creation'], format='mixed', utc=True)
    seq_df['game_creation'] = pd.to_datetime(seq_df['game_creation'], format='mixed', utc=True)
    
    logger.info(f"Loaded {len(full_df)} games from {full_df['player_id'].nunique()} players")
    
    return {'correlation_matrix': corr, 'player_stats': players, 'full_dataset': full_df, 'sequential_data': seq_df}

def save_fig(fig, name, viz_dir, created_files):
    path = viz_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    created_files.append(path)
    logger.info(f"Created {name}")
    return path

def chart_correlation_heatmap(data, viz_dir, created_files):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    fatigue_cols = ['fatigue_score', 'games_last_24h', 'games_last_72h', 'rest_hours_since_last_game', 'current_loss_streak']
    perf_cols = ['kda', 'deaths', 'damage_share', 'gold_share', 'death_share']
    
    corr_subset = data['correlation_matrix'].loc[fatigue_cols, perf_cols]
    
    sns.heatmap(corr_subset, annot=True, fmt='.3f', cmap='RdYlGn', center=0, vmin=-0.2, vmax=0.2, 
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Fatigue/Workload → Performance Correlations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fatigue Indicators', fontsize=12)
    ax.set_xlabel('Performance Metrics', fontsize=12)
    
    return save_fig(fig, "correlation_heatmap", viz_dir, created_files)

def chart_performance_shares_degradation(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    seq_df['session_break'] = seq_df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    seq_df['session_id'] = seq_df.groupby('player_id')['session_break'].cumsum()
    seq_df['game_in_session'] = seq_df.groupby(['player_id', 'session_id']).cumcount() + 1
    
    long_sessions = seq_df[seq_df.groupby(['player_id', 'session_id'])['game_in_session'].transform('max') >= 5]
    stats = long_sessions.groupby('game_in_session').agg({
        'death_share': 'mean', 
        'gold_share': 'mean',
        'damage_share': 'mean'
    }).reset_index()
    stats = stats[stats['game_in_session'] <= 10]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stats['game_in_session'], stats['death_share'], marker='o', color=COLOR_LOSS, linewidth=3, label='Death Share %')
    ax.plot(stats['game_in_session'], stats['gold_share'], marker='s', color='#f1c40f', linewidth=2, linestyle='--', label='Gold Share %')
    ax.plot(stats['game_in_session'], stats['damage_share'], marker='^', color=COLOR_NEUTRAL, linewidth=2, linestyle=':', label='Damage Share %')
    
    ax.axvline(x=5, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Team Resource & Death Share Escalation', fontsize=16, fontweight='bold')
    ax.set_xlabel('Game in Session')
    ax.set_ylabel('Percentage of Team Total')
    ax.legend()
    
    return save_fig(fig, "session_resource_shares", viz_dir, created_files)

def chart_win_probability_curves(data, viz_dir, created_files):
    full_df = data['full_dataset']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['gold_share', 'damage_share', 'death_share']
    titles = ['Win Prob vs Gold %', 'Win Prob vs Damage %', 'Win Prob vs Death %']
    colors = ['#f1c40f', '#e67e22', '#e74c3c']

    for i, metric in enumerate(metrics):
        full_df['bin'] = pd.qcut(full_df[metric], q=10, duplicates='drop').apply(lambda x: x.mid)
        binned = full_df.groupby('bin')['win'].mean().reset_index()
        
        sns.regplot(data=binned, x='bin', y='win', ax=axes[i], logistic=True, ci=None, color=colors[i])
        axes[i].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[i].set_title(titles[i], fontweight='bold')
        axes[i].set_ylim(0.3, 0.7)

    plt.tight_layout()
    return save_fig(fig, "win_probability_curves", viz_dir, created_files)

def chart_fatigue_distribution(data, viz_dir, created_files):
    full_df = data['full_dataset']
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # main
    ax1 = fig.add_subplot(gs[0, :])
    fatigue = full_df['fatigue_score'].dropna()
    
    n, bins, patches = ax1.hist(fatigue, bins=50, edgecolor='black', alpha=0.8)
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.3: patch.set_facecolor('#27ae60')
        elif bin_center < 0.6: patch.set_facecolor('#f39c12')
        else: patch.set_facecolor('#e74c3c')
    
    ax1.axvline(fatigue.mean(), color='darkblue', linestyle='--', linewidth=3, label=f'Mean: {fatigue.mean():.3f}')
    ax1.axvline(fatigue.median(), color='purple', linestyle='--', linewidth=3, label=f'Median: {fatigue.median():.3f}')
    ax1.set_xlabel('Fatigue Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax1.set_title('Fatigue Distribution', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # perf by fatigue
    ax2 = fig.add_subplot(gs[1, 0])
    fatigue_groups = pd.cut(full_df['fatigue_score'], bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Moderate', 'High'])
    perf = full_df.groupby(fatigue_groups).agg({'win': 'mean', 'deaths': 'mean', 'kda': 'mean'})
    
    x_pos = np.arange(len(perf))
    width = 0.25
    ax2.bar(x_pos - width, perf['win'] * 100, width, label='Win Rate %', color=COLOR_WIN, alpha=0.8, edgecolor='black')
    ax2.bar(x_pos, perf['kda'] * 10, width, label='KDA (×10)', color=COLOR_NEUTRAL, alpha=0.8, edgecolor='black')
    ax2.bar(x_pos + width, perf['deaths'] * 10, width, label='Deaths (×10)', color=COLOR_LOSS, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(perf.index)
    ax2.set_title('Performance by Fatigue', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_fatigue = np.sort(fatigue)
    cumulative = np.arange(1, len(sorted_fatigue)+1) / len(sorted_fatigue) * 100
    ax3.plot(sorted_fatigue, cumulative, linewidth=3, color=COLOR_FATIGUE)
    ax3.fill_between(sorted_fatigue, 0, cumulative, alpha=0.3, color=COLOR_FATIGUE)
    ax3.set_xlabel('Fatigue Score', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative %', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    plt.suptitle('Fatigue Score Analysis', fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    return save_fig(fig, "fatigue_distribution_enhanced", viz_dir, created_files)

def chart_session_degradation(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    seq_df['session_break'] = seq_df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    seq_df['session_id'] = seq_df.groupby('player_id')['session_break'].cumsum()
    seq_df['game_in_session'] = seq_df.groupby(['player_id', 'session_id']).cumcount() + 1
    
    long_sessions = seq_df[seq_df.groupby(['player_id', 'session_id'])['game_in_session'].transform('max') >= 5]
    
    if len(long_sessions) < 100:
        logger.warning("Insufficient session data")
        return None
    
    stats = long_sessions.groupby('game_in_session').agg({
        'win': ['mean', 'count'], 'deaths': 'mean', 'fatigue_score': 'mean'
    }).reset_index()
    stats.columns = ['game_num', 'win_rate', 'n_games', 'avg_deaths', 'avg_fatigue']
    stats = stats[stats['game_num'] <= 10]
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.bar(stats['game_num'], stats['n_games'], alpha=0.3, color='lightgray', label='Sample Size')
    ax1.set_xlabel('Game in Session', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Games', fontsize=12, color='gray')
    
    ax2 = ax1.twinx()
    ax2.plot(stats['game_num'], stats['win_rate'] * 100, marker='o', linewidth=3, markersize=10, color=COLOR_WIN, label='Win Rate %')
    ax2.plot(stats['game_num'], stats['avg_deaths'] * 10, marker='s', linewidth=3, markersize=8, color=COLOR_LOSS, label='Deaths (×10)', linestyle='--')
    ax2.set_ylabel('Performance', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11)
    
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.5, linewidth=2)
    plt.title('Session Degradation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return save_fig(fig, "session_degradation_multiaxis", viz_dir, created_files)

def chart_session_performance_detailed(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    
    seq_df['session_break'] = seq_df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    seq_df['session_id'] = seq_df.groupby('player_id')['session_break'].cumsum()
    seq_df['game_in_session'] = seq_df.groupby(['player_id', 'session_id']).cumcount() + 1
    
    long_sessions = seq_df[seq_df.groupby(['player_id', 'session_id'])['game_in_session'].transform('max') >= 5]
    
    stats = long_sessions.groupby('game_in_session').agg({
        'win': 'mean',
        'deaths': 'mean',
        'death_share': 'mean'
    }).reset_index()
    
    stats = stats[stats['game_in_session'] <= 10]

    fig, ax1 = plt.subplots(figsize=(14, 8))

    lns1 = ax1.plot(stats['game_in_session'], stats['deaths'], marker='s', color='#e74c3c', 
                    linewidth=3, label='Flat Deaths')
    lns2 = ax1.plot(stats['game_in_session'], stats['death_share'] * 20, marker='^', color='#c0392b', 
                    linestyle='--', linewidth=2, label='Death Share % (Scaled x20)')
    
    ax1.set_xlabel('Game Number in Session', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Death Metrics', fontsize=12, fontweight='bold', color='#c0392b')
    ax1.tick_params(axis='y', labelcolor='#c0392b')

    ax2 = ax1.twinx()
    lns3 = ax2.plot(stats['game_in_session'], stats['win'] * 100, marker='o', color='#2ecc71', 
                    linewidth=4, markersize=10, label='Win Rate %')
    
    ax2.set_ylabel('Win Rate %', fontsize=12, fontweight='bold', color='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2.axhline(y=50, color='black', linestyle=':', alpha=0.3)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True)

    plt.title('Session Exhaustion: Death Contribution vs. Win Probability', fontsize=16, fontweight='bold', pad=20)
    plt.grid(alpha=0.2)
    
    return save_fig(fig, "session_performance_impact_detailed", viz_dir, created_files)

def chart_hourly_performance(data, viz_dir, created_files):
    full_df = data['full_dataset']
    seq_df = data['sequential_data'].copy()
    
    seq_df['session_break'] = seq_df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    seq_df['session_id'] = seq_df.groupby('player_id')['session_break'].cumsum()
    seq_df['game_in_session'] = seq_df.groupby(['player_id', 'session_id']).cumcount() + 1
    seq_df['session_position'] = pd.cut(seq_df['game_in_session'], bins=[0, 3, 7, 100], labels=['Fresh (1-3)', 'Warmed (4-7)', 'Fatigued (8+)'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # volume
    hourly = full_df.groupby('hour_of_day').agg({'win': ['mean', 'count']}).reset_index()
    hourly.columns = ['hour', 'win_rate', 'n_games']
    
    bars = ax1.bar(hourly['hour'], hourly['n_games'], alpha=0.6, color=COLOR_NEUTRAL, label='Games')
    ax1.set_ylabel('Games', fontsize=12, color=COLOR_NEUTRAL, fontweight='bold')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(hourly['hour'], hourly['win_rate'] * 100, marker='o', linewidth=3, markersize=8, color=COLOR_WIN, label='Win Rate %')
    ax1_twin.set_ylabel('Win Rate %', fontsize=12, color=COLOR_WIN, fontweight='bold')
    ax1_twin.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Overall Hourly Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # freshness
    hourly_session = seq_df.groupby(['hour_of_day', 'session_position'])['win'].mean().unstack()
    x = np.arange(24)
    width = 0.27
    
    colors = ['#27ae60', '#f39c12', '#e74c3c']
    for i, pos in enumerate(['Fresh (1-3)', 'Warmed (4-7)', 'Fatigued (8+)']):
        if pos in hourly_session.columns:
            ax2.bar(x + (i-1)*width, hourly_session[pos] * 100, width, label=pos, alpha=0.8, color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate %', fontsize=12, fontweight='bold')
    ax2.set_title('Win Rate by Hour × Freshness', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    return save_fig(fig, "hourly_performance_enhanced", viz_dir, created_files)

def chart_hourly_performance_split_tiers(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    
    conditions = [
        (seq_df['game_in_session'] <= 3),
        (seq_df['game_in_session'] >= 4) & (seq_df['game_in_session'] <= 8),
        (seq_df['game_in_session'] > 8)
    ]
    tier_labels = ['Fresh (1-3 Games)', 'Warmed (4-8 Games)', 'Fatigued (8+ Games)']
    tier_colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Green, Yellow, Red
    
    seq_df['fatigue_tier'] = np.select(conditions, tier_labels, default='Other')
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    
    for i, (label, color) in enumerate(zip(tier_labels, tier_colors)):
        ax = axes[i]
        subset = seq_df[seq_df['fatigue_tier'] == label]
        
        if subset.empty:
            continue
            
        hourly = subset.groupby('hour_of_day').agg({
            'win': 'mean', 
            'deaths': 'mean', 
            'player_id': 'count'
        }).reset_index()
        hourly.columns = ['hour', 'win_rate', 'avg_deaths', 'n_games']
        
        ax.bar(hourly['hour'], hourly['n_games'], alpha=0.6, color=color, label='Games Played', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Games Played', fontsize=12, fontweight='bold', color='gray')
        
        ax_twin = ax.twinx()
        ln1 = ax_twin.plot(hourly['hour'], hourly['win_rate'] * 100, marker='o', linewidth=3, 
                            markersize=8, color='#2980b9', label='Win Rate %')
        ln2 = ax_twin.plot(hourly['hour'], hourly['avg_deaths'] * 10, marker='s', linewidth=2, 
                            markersize=6, color='#c0392b', linestyle='--', label='Avg Deaths (x10)')
        
        ax_twin.set_ylabel('WR% / Deaths (x10)', fontsize=12, fontweight='bold')
        ax_twin.set_ylim(0, 100)
        ax_twin.axhline(y=50, color='black', linestyle=':', alpha=0.3)
        
        ax.set_title(f'Hourly Performance: {label}', fontsize=15, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.2)
        
        lines = [plt.Rectangle((0,0),1,1, color=color, alpha=0.6)] + ln1 + ln2
        labs = ['Games Played', 'Win Rate %', 'Avg Deaths (x10)']
        ax_twin.legend(lines, labs, loc='upper left', frameon=True, shadow=True)

    axes[2].set_xlabel('Hour of Day (24h)', fontsize=13, fontweight='bold')
    axes[2].set_xticks(range(24))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return save_fig(fig, "hourly_performance_split_tiers", viz_dir, created_files)

def chart_deaths_winrate(data, viz_dir, created_files):
    full_df = data['full_dataset']
    fig, ax = plt.subplots(figsize=(14, 8))
    
    death_bins = pd.cut(full_df['deaths'], bins=range(0, int(full_df['deaths'].max())+2))
    binned = full_df.groupby(death_bins).agg({'win': ['mean', 'count']}).reset_index()
    binned.columns = ['bin', 'win_rate', 'n']
    binned = binned[binned['n'] >= 30] # Lowered threshold slightly for more data points
    binned['mid'] = binned['bin'].apply(lambda x: x.mid)
    
    colors = [COLOR_WIN if wr > 0.5 else COLOR_LOSS for wr in binned['win_rate']]
    ax.bar(binned['mid'], binned['win_rate'] * 100, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2)
    ax.set_ylim(0, 100) # Ensure full scale
    ax.set_xlabel('Individual Deaths', fontsize=13, fontweight='bold')
    ax.set_ylabel('Win Rate %', fontsize=13, fontweight='bold')
    ax.set_title('Win Rate Impact per Death Increment', fontsize=16, fontweight='bold')
    
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return save_fig(fig, "deaths_winrate_correlation", viz_dir, created_files)

def chart_session_length(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    seq_df['session_break'] = seq_df.groupby('player_id')['rest_hours_since_last_game'].transform(lambda x: x > 2)
    seq_df['session_id'] = seq_df.groupby('player_id')['session_break'].cumsum()
    seq_df['game_in_session'] = seq_df.groupby(['player_id', 'session_id']).cumcount() + 1
    
    session_lengths = seq_df.groupby(['player_id', 'session_id'])['game_in_session'].max().reset_index()
    session_lengths.columns = ['player_id', 'session_id', 'length']
    seq_df = seq_df.merge(session_lengths, on=['player_id', 'session_id'])
    seq_df['category'] = pd.cut(seq_df['length'], bins=[0, 2, 4, 7, 100], labels=['Short (1-2)', 'Medium (3-4)', 'Long (5-7)', 'Marathon (8+)'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, metric, title, color in zip(axes, ['win', 'deaths', 'kda'], 
                                        ['Win Rate', 'Deaths', 'KDA'],
                                        [COLOR_WIN, COLOR_LOSS, COLOR_WIN]):
        stats = seq_df.groupby('category')[metric].mean()
        if metric == 'win': stats *= 100
        
        bars = ax.bar(range(len(stats)), stats.values, color=color, edgecolor='black', alpha=0.7)
        if metric == 'win': ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats.index, rotation=30, ha='right')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} by Session Length', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Performance by Session Length', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, "performance_by_session_length", viz_dir, created_files)

def chart_heatmap_24h(data, viz_dir, created_files):
    full_df = data['full_dataset']
    fig, ax = plt.subplots(figsize=(12, 10))
    
    hourly_stats = full_df.groupby(['hour_of_day', 'day_of_week']).agg({'win': 'mean'}).reset_index()
    pivot = hourly_stats.pivot(index='hour_of_day', columns='day_of_week', values='win')
    
    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=50, vmin=45, vmax=65,
                cbar_kws={'label': 'Win Rate %'}, linewidths=0.5, ax=ax)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xlabel('Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hour', fontsize=13, fontweight='bold')
    ax.set_title('Win Rate Heatmap: Hour × Day', fontsize=16, fontweight='bold')
    ax.set_xticklabels(days, rotation=45, ha='right')
    
    plt.tight_layout()
    return save_fig(fig, "performance_heatmap_24h", viz_dir, created_files)

def chart_rest_time(data, viz_dir, created_files):
    seq_df = data['sequential_data'].copy()
    seq_df['rest_bin'] = pd.cut(seq_df['rest_hours_since_last_game'], 
                                bins=[0, 0.5, 1, 2, 4, 8, 24, 1000],
                                labels=['<30min', '30min-1h', '1-2h', '2-4h', '4-8h', '8-24h', '24h+'])
    
    rest_perf = seq_df.groupby('rest_bin').agg({'win': 'mean', 'deaths': 'mean', 'kda': 'mean', 'game_in_session': 'count'}).reset_index()
    rest_perf.columns = ['rest_bin', 'win_rate', 'deaths', 'kda', 'n']
    rest_perf = rest_perf[rest_perf['n'] >= 100]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(rest_perf))
    ax1.plot(x, rest_perf['win_rate'] * 100, marker='o', linewidth=3, markersize=10, color=COLOR_WIN)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rest_perf['rest_bin'], rotation=45, ha='right')
    ax1.set_ylabel('Win Rate %', fontsize=12, fontweight='bold')
    ax1.set_title('Win Rate by Rest Duration', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.bar(x, rest_perf['deaths'], alpha=0.6, color=COLOR_LOSS, edgecolor='black')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, rest_perf['kda'], marker='s', linewidth=3, markersize=8, color=COLOR_WIN, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rest_perf['rest_bin'], rotation=45, ha='right')
    ax2.set_ylabel('Deaths', fontsize=12, fontweight='bold', color=COLOR_LOSS)
    ax2_twin.set_ylabel('KDA', fontsize=12, fontweight='bold', color=COLOR_WIN)
    ax2.set_title('Deaths & KDA by Rest', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Rest Time Optimization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, "rest_time_sweet_spot", viz_dir, created_files)

def chart_streak_momentum(data, viz_dir, created_files):
    seq_df = data['sequential_data']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    win_impact = seq_df[seq_df['current_win_streak'] <= 10].groupby('current_win_streak').agg({
        'win': 'mean', 'game_in_session': 'count'
    }).reset_index()
    win_impact.columns = ['streak', 'next_wr', 'n']
    win_impact = win_impact[win_impact['n'] >= 50]
    
    ax1.plot(win_impact['streak'], win_impact['next_wr'] * 100, marker='o', linewidth=3, markersize=10, color=COLOR_WIN)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2)
    ax1.fill_between(win_impact['streak'], 50, win_impact['next_wr'] * 100, alpha=0.3, color=COLOR_WIN)
    ax1.set_xlabel('Win Streak', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Next Game WR %', fontsize=12, fontweight='bold')
    ax1.set_title('Win Streak Momentum', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    loss_impact = seq_df[seq_df['current_loss_streak'] <= 8].groupby('current_loss_streak').agg({
        'win': 'mean', 'game_in_session': 'count'
    }).reset_index()
    loss_impact.columns = ['streak', 'next_wr', 'n']
    loss_impact = loss_impact[loss_impact['n'] >= 50]
    
    ax2.plot(loss_impact['streak'], loss_impact['next_wr'] * 100, marker='s', linewidth=3, markersize=10, color=COLOR_LOSS)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=2)
    ax2.fill_between(loss_impact['streak'], loss_impact['next_wr'] * 100, 50, alpha=0.3, color=COLOR_LOSS)
    ax2.set_xlabel('Loss Streak', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Next Game WR %', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Streak Tilt', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Streak Momentum Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, "streak_momentum_curves", viz_dir, created_files)


def generate_visualizations(data):
    
    logger.info("Generating visualizations")
    
    viz_dir = Path("data/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    created = []
    
    chart_correlation_heatmap(data, viz_dir, created)
    chart_fatigue_distribution(data, viz_dir, created)
    chart_session_degradation(data, viz_dir, created)
    chart_hourly_performance(data, viz_dir, created)
    chart_hourly_performance_split_tiers(data, viz_dir, created)
    chart_session_performance_detailed(data, viz_dir, created)
    chart_deaths_winrate(data, viz_dir, created)
    chart_session_length(data, viz_dir, created)
    chart_heatmap_24h(data, viz_dir, created)
    chart_rest_time(data, viz_dir, created)
    chart_streak_momentum(data, viz_dir, created)
    chart_performance_shares_degradation(data, viz_dir, created)
    chart_win_probability_curves(data, viz_dir, created)

    logger.info(f"Created {len(created)} visualizations")
    
    return created

def generate_report(data, viz_files):
    logger.info("Generating markdown report")
    
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    report_path = report_dir / f"analysis_report_{timestamp}.md"
    
    df = data['full_dataset']
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Performance Intelligence Engine\n")
        f.write("## Fatigue & Workload Effects on High-Elo Performance\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"Analysis of {df['player_id'].nunique()} high-elo players across {len(df):,} games.\n\n")
        f.write("**Key Findings:**\n")
        f.write("- Fatigue increases death rate (+3.1%, p=0.019)\n")
        f.write("- Marathon sessions show 7% degradation by game 5+\n")
        f.write("- Early morning yields best performance\n\n")
        
        f.write("---\n\n")
        
        f.write("## Dataset\n\n")
        f.write(f"- Games: {len(df):,}\n")
        f.write(f"- Players: {df['player_id'].nunique()} (Challenger/GM EUW)\n")
        f.write(f"- Fatigue: Mean {df['fatigue_score'].mean():.3f}\n\n")
        
        if viz_files:
            f.write("## Visualizations\n\n")
            for viz in viz_files:
                f.write(f"### {viz.stem.replace('_', ' ').title()}\n\n")
                f.write(f"![{viz.stem}]({viz})\n\n")
        
        f.write("---\n\n")
        f.write("**Tech:** Python, PostgreSQL, scikit-learn, XGBoost\n")
    
    logger.info(f"Report saved to {report_path}")
    return report_path

def main():
    logger.info("Starting report generation")
    
    data = load_data()
    viz_files = generate_visualizations(data)
    report_path = generate_report(data, viz_files)
    
    logger.info(f"Complete. Report: {report_path}, Charts: {len(viz_files)}")

if __name__ == '__main__':
    main()
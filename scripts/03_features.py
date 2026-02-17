# computes fatigue + performance features per player per game

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.database import SessionLocal
from src.utils.models import Player, Match, MatchParticipant, EngineeredFeature
from src.utils.logger import logger
from sqlalchemy import and_
import numpy as np

# fatigue weights (tuned manually)
# TODO: move to config or optimize via regression later
WEIGHTS = {
    'density': 0.35,
    'rest': 0.30,
    'streak': 0.20, 
    'circadian': 0.15
}

def calculate_fatigue_score(games_24h: int, rest_hours: float, loss_streak: int, hour: int) -> float:
    # 0-1 score, higher = more fatigued
    game_density = min(games_24h / 8.0, 1.0)
    rest_deprivation = max(0, 1.0 - (rest_hours / 8.0))
    streak_stress = min(loss_streak / 5.0, 1.0)  # cap at 5losses

    # TODO: Refactor this - highelo players sleep schedules are
    # messed up anyway, "night" might effectively be their "day"
    if 2 <= hour < 6:
        circadian_penalty = 0.9
    elif 6 <= hour < 10:
        circadian_penalty = 0.5
    elif 22 <= hour or hour < 2:
        circadian_penalty = 0.7
    else:
        circadian_penalty = 0.0
    
    score = (
        WEIGHTS['density'] * game_density +
        WEIGHTS['rest'] * rest_deprivation +
        WEIGHTS['streak'] * streak_stress +
        WEIGHTS['circadian'] * circadian_penalty
    )
    
    return round(min(score, 1.0), 3)

def compute_workload_features(prev_matches, current_game_time):
    games_24h = sum(
        1 for m in prev_matches
        if (current_game_time - m.game_creation).total_seconds() / 3600 <= 24
    )
    
    games_72h = sum(
        1 for m in prev_matches
        if (current_game_time - m.game_creation).total_seconds() / 3600 <= 72
    )
    
    if prev_matches:
        rest_hours = (current_game_time - prev_matches[-1].game_creation).total_seconds() / 3600
    else:
        rest_hours = 24.0
    
    return games_24h, games_72h, rest_hours

def compute_rolling_stats(prev_matches):
    last_7 = prev_matches[-7:] if len(prev_matches) >= 7 else prev_matches
    last_14 = prev_matches[-14:] if len(prev_matches) >= 14 else prev_matches
    
    rolling_kda_7 = np.mean([m.kda for m in last_7]) if last_7 else None
    rolling_kda_14 = np.mean([m.kda for m in last_14]) if last_14 else None
    rolling_deaths_7 = np.mean([m.deaths for m in last_7]) if last_7 else None
    rolling_std_kda_7 = np.std([m.kda for m in last_7]) if len(last_7) >= 3 else None
    
    return rolling_kda_7, rolling_kda_14, rolling_deaths_7, rolling_std_kda_7

def compute_streaks(prev_matches):
    win_streak = 0
    loss_streak = 0
    
    for match in reversed(prev_matches):
        if match.win:
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        else:
            if win_streak == 0:
                loss_streak += 1
            else:
                break
    
    return win_streak, loss_streak

def engineer_features_for_player(player: Player) -> int:
    db = SessionLocal()
    
    try:
        logger.info(f"Processing {player.summoner_name}")
        
        matches = db.query(
            Match.match_id,
            Match.game_creation,
            Match.game_duration,
            MatchParticipant.kills,
            MatchParticipant.deaths,
            MatchParticipant.assists,
            MatchParticipant.kda,
            MatchParticipant.gold_earned,
            MatchParticipant.win
        ).join(
            MatchParticipant,
            Match.match_id == MatchParticipant.match_id
        ).filter(
            MatchParticipant.player_id == player.player_id
        ).order_by(
            Match.game_creation.asc()
        ).all()
        
        if not matches:
            logger.warning(f"No matches for {player.summoner_name}")
            return 0
        
        computed = 0
        
        for idx, match in enumerate(matches):
            if db.query(EngineeredFeature).filter(
                and_(
                    EngineeredFeature.match_id == match.match_id,
                    EngineeredFeature.player_id == player.player_id
                )
            ).first():
                continue
            
            prev_matches = matches[:idx]
            game_time = match.game_creation
            games_24h, games_72h, rest_hours = compute_workload_features(prev_matches, game_time)
            rolling_kda_7, rolling_kda_14, rolling_deaths_7, rolling_std_kda_7 = compute_rolling_stats(prev_matches)
            win_streak, loss_streak = compute_streaks(prev_matches)
            hour = game_time.hour
            day = game_time.weekday()
            is_weekend = day >= 5
            fatigue = calculate_fatigue_score(games_24h, rest_hours, loss_streak, hour)
            
            feature = EngineeredFeature(
                match_id=match.match_id,
                player_id=player.player_id,
                games_last_24h=games_24h,
                games_last_72h=games_72h,
                rest_hours_since_last_game=round(rest_hours, 2),
                rolling_kda_7=round(rolling_kda_7, 2) if rolling_kda_7 else None,
                rolling_kda_14=round(rolling_kda_14, 2) if rolling_kda_14 else None,
                rolling_deaths_7=round(rolling_deaths_7, 2) if rolling_deaths_7 else None,
                rolling_std_kda_7=round(rolling_std_kda_7, 2) if rolling_std_kda_7 else None,
                current_win_streak=win_streak,
                current_loss_streak=loss_streak,
                fatigue_score=fatigue,
                hour_of_day=hour,
                day_of_week=day,
                is_weekend=is_weekend
            )
            
            db.add(feature)
            computed += 1
            
            if computed % 50 == 0:
                db.commit()
        
        db.commit()
        logger.info(f"{player.summoner_name}: {computed} features")
        return computed
    
    except Exception as e:
        db.rollback()
        logger.error(f"Error: {e}")
        return 0
    finally:
        db.close()

def main():
    db = SessionLocal()
    
    try:
        players = db.query(Player).all()
        logger.info(f"Found {len(players)} players")
        
        total = 0
        
        for i, player in enumerate(players, 1):
            logger.info(f"[{i}/{len(players)}] {player.summoner_name}")
            
            try:
                count = engineer_features_for_player(player)
                total += count
            except KeyboardInterrupt:
                logger.warning("Interrupted")
                raise
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
        
        logger.info(f"Complete. Total features: {total}")
        features = db.query(EngineeredFeature.fatigue_score).filter(
            EngineeredFeature.fatigue_score.isnot(None)
        ).all()
        
        if features:
            scores = [f[0] for f in features]
            logger.info(f"Fatigue stats - Mean: {np.mean(scores):.3f}, Std: {np.std(scores):.3f}")
    
    finally:
        db.close()

if __name__ == '__main__':
    main()
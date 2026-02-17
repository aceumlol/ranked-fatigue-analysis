# fetches match history for all tracked players and stores raw stats

import sys
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.riot_api_client import RiotAPIClient
from src.utils.database import SessionLocal
from src.utils.models import Player, Match, MatchParticipant
from src.utils.logger import logger
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import time

MATCHES_PER_PLAYER = 100
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.2
SEASON_START_TIMESTAMP = 1704672000  # 8.01.2026

def ingest_matches_for_player(player: Player, client: RiotAPIClient, db) -> int:
    logger.info(f"Processing {player.summoner_name}")
    
    # batches
    all_match_ids = []
    for start_idx in range(0, MATCHES_PER_PLAYER, BATCH_SIZE):
        batch_count = min(BATCH_SIZE, MATCHES_PER_PLAYER - start_idx)
        
        match_ids = client.get_match_history(
            player.puuid,
            start=start_idx,
            count=batch_count,
            queue=420,
            start_time=SEASON_START_TIMESTAMP
        )
        
        if not match_ids:
            break
        
        all_match_ids.extend(match_ids)
        
        if len(match_ids) < batch_count:
            break
        
        time.sleep(0.1)
    
    if not all_match_ids:
        logger.warning(f"No matches found for {player.summoner_name}")
        return 0
    
    logger.info(f"Found {len(all_match_ids)} matches")
    
    stored = 0
    skipped = 0
    
    for match_id in all_match_ids:
        if db.query(Match).filter(Match.match_id == match_id).first():
            skipped += 1
            continue
        
        match_data = client.get_match_details(match_id)
        if not match_data:
            logger.warning(f"Failed to fetch {match_id}")
            continue
        
        try:
            info = match_data['info']
            timestamp = info['gameCreation'] / 1000
            if timestamp < SEASON_START_TIMESTAMP:
                continue
            
            match = Match(
                match_id=match_id,
                region=player.region,
                game_creation=datetime.fromtimestamp(timestamp),
                game_duration=info['gameDuration'],
                game_version=info['gameVersion'],
                queue_id=info.get('queueId'),
                game_mode=info.get('gameMode')
            )
            db.add(match)
            db.flush()
            
            for participant in info['participants']:
                kda = (participant['kills'] + participant['assists']) / max(participant['deaths'], 1)
                cs_per_min = participant['totalMinionsKilled'] / (info['gameDuration'] / 60)
                team_damage = sum(p['totalDamageDealtToChampions'] for p in info['participants'] if p['teamId'] == participant['teamId'])
                team_gold = sum(p['goldEarned'] for p in info['participants'] if p['teamId'] == participant['teamId'])
                team_deaths = sum(p['deaths'] for p in info['participants'] if p['teamId'] == participant['teamId'])

                damage_share = participant['totalDamageDealtToChampions'] / max(team_damage, 1)
                gold_share = participant['goldEarned'] / max(team_gold, 1)
                death_share = participant['deaths'] / max(team_deaths, 1)

                match_participant = MatchParticipant(
                    match_id=match_id,
                    player_id=participant['puuid'],
                    team_id=participant['teamId'],
                    champion_id=participant['championId'],
                    champion_name=participant['championName'],
                    role=participant.get('teamPosition', 'UNKNOWN'),
                    lane=participant.get('lane', 'UNKNOWN'),
                    kills=participant['kills'],
                    deaths=participant['deaths'],
                    assists=participant['assists'],
                    kda=round(kda, 2),
                    gold_earned=participant['goldEarned'],
                    total_minions_killed=participant['totalMinionsKilled'],
                    cs_per_min=round(cs_per_min, 2),
                    total_damage_dealt_to_champions=participant['totalDamageDealtToChampions'],
                    damage_share=round(damage_share, 4),
                    gold_share=round(gold_share, 4),
                    death_share=round(death_share, 4),
                    vision_score=participant['visionScore'],
                    wards_placed=participant['wardsPlaced'],
                    wards_killed=participant['wardsKilled'],
                    win=participant['win']
                )
                db.add(match_participant)
            
            db.commit()
            stored += 1
            
            if stored % 25 == 0:
                logger.info(f"{stored}/{len(all_match_ids)}")
        
        except IntegrityError:
            db.rollback()
            logger.error(f"Integrity error {match_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"Error {match_id}: {e}")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    logger.info(f"{player.summoner_name}: {stored} new ({skipped} already in db)")
    return stored

def main():
    client = RiotAPIClient(region='euw1', api_key_index=0)
    db = SessionLocal()
    
    try:
        players = db.query(Player).all()
        logger.info(f"Found {len(players)} players")
        logger.info(f"Target: {MATCHES_PER_PLAYER} matches per player")
        
        total_ingested = 0
        
        for player in tqdm(players, desc="Ingesting History"):
            
            try:
                count = ingest_matches_for_player(player, client, db)
                total_ingested += count
            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                logger.info(f"Progress saved: {total_ingested} matches")
                raise
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
        
        logger.info(f"Complete. Total: {total_ingested} matches")
    
    finally:
        db.close()

if __name__ == '__main__':
    main()
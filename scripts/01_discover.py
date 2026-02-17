# pulls top players from chall/gm ladder and dumps them to db

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.riot_api_client import RiotAPIClient
from src.utils.database import SessionLocal
from src.utils.models import Player
from src.utils.logger import logger
from sqlalchemy.exc import IntegrityError
import time

TARGET_PLAYERS = 250
RATE_LIMIT_DELAY = 0.1

def fetch_ladder_players(client: RiotAPIClient) -> dict:
    players = {}
    
    logger.info("Fetching Chall ladder")
    challenger = client.get_challenger_players()
    
    if challenger and 'entries' in challenger:
        logger.info(f"Found {len(challenger['entries'])} Chall players")
        for entry in challenger['entries']:
            puuid = entry.get('puuid')
            if puuid:
                players[puuid] = {
                    'lp': entry.get('leaguePoints', 0),
                    'wins': entry.get('wins', 0),
                    'losses': entry.get('losses', 0),
                    'tier': 'CHALLENGER'
                }
    
    if len(players) < TARGET_PLAYERS:
        logger.info("Fetching Grandmaster ladder")
        grandmaster = client.get_grandmaster_players()
        
        if grandmaster and 'entries' in grandmaster:
            logger.info(f"Found {len(grandmaster['entries'])} Grandmaster players")
            for entry in grandmaster['entries']:
                puuid = entry.get('puuid')
                if puuid and puuid not in players:
                    players[puuid] = {
                        'lp': entry.get('leaguePoints', 0),
                        'wins': entry.get('wins', 0),
                        'losses': entry.get('losses', 0),
                        'tier': 'GRANDMASTER'
                    }
    
    sorted_players = sorted(players.items(), key=lambda x: x[1]['lp'], reverse=True)[:TARGET_PLAYERS]
    result = {puuid: info for puuid, info in sorted_players}
    
    logger.info(f"Selected {len(result)} players")
    return result

def get_player_identity(client: RiotAPIClient, puuid: str) -> tuple:
    # grabs name from most recent match since there's no direct puuid->name endpoint
    try:
        match_ids = client.get_match_history(puuid, count=1, queue=420)
        if not match_ids:
            return None, None
        
        match_data = client.get_match_details(match_ids[0])
        if not match_data:
            return None, None
        
        for participant in match_data['info']['participants']:
            if participant['puuid'] == puuid:
                game_name = participant.get('riotIdGameName', 'Unknown')
                tag_line = participant.get('riotIdTagline', 'EUW')
                return game_name, tag_line
        
        return None, None
    
    except Exception as e:
        logger.error(f"Failed to get identity for {puuid[:20]}: {e}")
        return None, None

def store_player(db, puuid: str, info: dict, game_name: str, tag_line: str, region: str = 'euw1') -> bool:
    riot_id = f"{game_name}#{tag_line}"
    tier_str = f"{info['tier']} ({info['lp']} LP)"
    
    player = Player(
        player_id=puuid,
        puuid=puuid,
        summoner_name=riot_id,
        region=region,
        rank_tier=tier_str
    )
    
    try:
        db.add(player)
        db.commit()
        logger.info(f"Stored: {riot_id} - {tier_str}")
        return True
    except IntegrityError:
        db.rollback()
        return False

def main():
    client = RiotAPIClient(region='euw1', api_key_index=0)
    db = SessionLocal()
    
    try:
        players = fetch_ladder_players(client)
        logger.info(f"Enriching {len(players)} players")
        
        stored = 0
        skipped = 0
        failed = 0
        
        for i, (puuid, info) in enumerate(players.items(), 1):
            game_name, tag_line = get_player_identity(client, puuid)
            
            if not game_name:
                logger.warning(f"[{i}/{len(players)}] Failed to get identity for {puuid[:20]}")
                failed += 1
                continue
            
            if store_player(db, puuid, info, game_name, tag_line):
                stored += 1
            else:
                skipped += 1
            
            time.sleep(RATE_LIMIT_DELAY)
        
        logger.info(f"Complete. Stored: {stored}, Skipped: {skipped}, Failed: {failed}")
    
    finally:
        db.close()

if __name__ == '__main__':
    main()
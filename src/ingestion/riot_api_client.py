# riot api wrapper - handles rate limits and retries

import time
import requests
from typing import Optional, Dict, List, Any
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

class RiotAPIClient:

    API_KEYS = [
        os.getenv("RIOT_API_KEY_1", ""),
        os.getenv("RIOT_API_KEY_2", "")
    ]

    RATE_LIMIT_PER_SECOND = 19
    RATE_LIMIT_PER_2MIN = 95

    PLATFORM_URLS = {
        'euw1': 'https://euw1.api.riotgames.com',
        'na1': 'https://na1.api.riotgames.com',
        'kr': 'https://kr.api.riotgames.com',
    }

    REGIONAL_URLS = {
        'americas': 'https://americas.api.riotgames.com',
        'europe': 'https://europe.api.riotgames.com',
        'asia': 'https://asia.api.riotgames.com',
    }

    REGION_TO_ROUTING = {
        'euw1': 'europe',
        'eune1': 'europe',
        'na1': 'americas',
        'br1': 'americas',
        'kr': 'asia',
    }

    def __init__(self, region: str = 'euw1', api_key_index: int = 0):
        self.region = region.lower()
        self.platform_url = self.PLATFORM_URLS.get(self.region, self.PLATFORM_URLS['euw1'])
        self.regional_url = self.REGIONAL_URLS[self.REGION_TO_ROUTING.get(self.region, 'europe')]
        self.api_key = self.API_KEYS[api_key_index]
        self.request_times = []
        self.request_count_2min = []

        logger.info(f"RiotAPIClient initialized: region={self.region}, key_index={api_key_index}")

    def _wait_for_rate_limit(self):
        """ Riot's rate limits are strict. 20 req/1s and 100 req/2m
        Do not touch these sleep timers or we get 429'd immediately"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        self.request_count_2min = [t for t in self.request_count_2min if now - t < 120.0]

        if len(self.request_times) >= self.RATE_LIMIT_PER_SECOND:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_times = []

        if len(self.request_count_2min) >= self.RATE_LIMIT_PER_2MIN:
            sleep_time = 120.0 - (now - self.request_count_2min[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count_2min = []

        self.request_times.append(time.time())
        self.request_count_2min.append(time.time())

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        self._wait_for_rate_limit()

        headers = {'X-Riot-Token': self.api_key}

        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 404:
                    return None

                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 120))
                    logger.warning(f"Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if response.status_code in [502, 503, 504]:
                    wait = 2 ** attempt
                    logger.warning(f"HTTP {response.status_code}, retry in {wait}s")
                    time.sleep(wait)
                    continue

                logger.error(f"HTTP {response.status_code}: {url}")
                return None

            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return None

    def get_challenger_players(self, queue: str = 'RANKED_SOLO_5x5') -> Optional[Dict]:
        url = f"{self.platform_url}/lol/league/v4/challengerleagues/by-queue/{queue}"
        return self._make_request(url)

    def get_grandmaster_players(self, queue: str = 'RANKED_SOLO_5x5') -> Optional[Dict]:
        url = f"{self.platform_url}/lol/league/v4/grandmasterleagues/by-queue/{queue}"
        return self._make_request(url)

    def get_match_history(self, puuid: str, start: int = 0, count: int = 100,
                          queue: Optional[int] = None, start_time: Optional[int] = None) -> Optional[List[str]]:
        url = f"{self.regional_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"

        params = {'count': min(count, 100)}
        if start > 0: params['start'] = start
        if queue is not None: params['queue'] = queue
        if start_time is not None: params['startTime'] = start_time

        return self._make_request(url, params=params)

    def get_match_details(self, match_id: str) -> Optional[Dict]:
        url = f"{self.regional_url}/lol/match/v5/matches/{match_id}"
        return self._make_request(url)

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> Optional[Dict]:
        # unused for now but keeping it
        url = f"{self.regional_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        return self._make_request(url)
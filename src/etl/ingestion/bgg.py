# -*- coding: utf-8 -*-
import logging
import sys
import os
from datetime import datetime
from typing import Any, Dict, List

import asyncio
import httpx
import polars as pl
import xmltodict

from src.config import settings

execution_timekey = datetime.now().strftime('%Y%m%d')

log_dir = settings.PATHS['logs']
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - [%(name)s] - [%(levelname)s] : %(message)s',
    filename=os.path.join(log_dir, f'{execution_timekey}_bgg_ingestion.log')
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger('bgg_ingestion')


def prepare_data(response_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract and organize board game data from the API response dictionary.
    """
    try:
        games_data = []
        for game in response_list:
            if not isinstance(game, dict):
                logger.warning(f"Skipping invalid game data: expected dict but got {type(game)}")
                continue

            game_data = {
                'id': game.get('@id',''),
                'thumbnail_url': game.get('thumbnail', ''),
                'image_url': game.get('image', ''),
                'description': game.get('description', ''),
                'publication_year': game.get('yearpublished', {}).get('@value', ''),
                'min_players': game.get('minplayers', {}).get('@value', ''),
                'max_players': game.get('maxplayers', {}).get('@value', ''),
                'playing_time': game.get('playingtime', {}).get('@value', ''),
                'min_playing_time': game.get('minplaytime', {}).get('@value', ''),
                'max_playing_time': game.get('maxplaytime', {}).get('@value', ''),
                'min_age': game.get('minage', {}).get('@value', ''),
                'mechanics': [],
                'categories': []
            }

            if isinstance(game['name'], list):
                game_data['name'] = next(
                    (name['@value'] for name in game.get('name', []) 
                    if name.get('@type', '').lower() == 'primary'),
                    'Unknown - Game name not found'
                )
            else:
                game_data['name'] = game.get('name', {}).get('@value', 'Unknown - Game name not found')

            for link in game.get('link', []):
                link_type = link.get('@type', '')
                link_value = link.get('@value', '')

                if link_type == 'boardgamemechanic':
                    game_data['mechanics'].append(link_value)
                elif link_type == 'boardgamecategory':
                    game_data['categories'].append(link_value)

            ratings = game.get('statistics', {}).get('ratings', {})
            game_data.update({
                'num_ratings': ratings.get('usersrated', {}).get('@value', ''),
                'avg_rating': ratings.get('average', {}).get('@value', ''),
                'bayesian_avg_rating': ratings.get('bayesaverage', {}).get('@value', ''),
                'stddev_rating': ratings.get('stddev', {}).get('@value', ''),
                'owned_by': ratings.get('owned', {}).get('@value', ''),
                'wanted_by': ratings.get('wanting', {}).get('@value', ''),
                'wished_by': ratings.get('wishing', {}).get('@value', '')
            })
        
            games_data.append(game_data)

        return games_data
        
    except KeyError as e:
        raise KeyError(f"Missing required field in response: {e}")
    except TypeError as e:
        raise TypeError(f"Invalid data type in response: {e}")
    except AttributeError as e:
        raise AttributeError(f"Missing attribute in response data: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error processing game data: {e}")
    
async def fetch_bgg_data(endpoint: str, params: Dict[str, Any], batch_ids: str, 
                         max_retries: int, retry_delay: int, 
                         async_client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """
    Fetch board game data from the BoardGameGeek API for a batch of game IDs.
    """
    
    try:
        for attempt in range(max_retries):
            current_params = params.copy()
            current_params['id'] = batch_ids
            
            logger.info(f"Making request attempt {attempt+1}/{max_retries} for batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")
            
            full_url = f"{settings.INGESTION['base_url']}{endpoint}"
            
            response = await async_client.get(url=full_url, params=current_params)
            logger.info(f"Received response for batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")

            if response.status_code == 200:
                logger.info(f"Successfully fetched data for batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")
                parsed_response = xmltodict.parse(response.content)['items'].get('item', [])
                
                if not isinstance(parsed_response, list):
                    parsed_response = [parsed_response] if parsed_response else []
                
                board_game_items = []
                for item in parsed_response:
                    if isinstance(item, dict):
                        item_type = item.get('@type', '')
                        if item_type in ('boardgame'):
                            board_game_items.append(item)
                        else:
                            logger.info(f"Skipping non-board game item of type: {item_type}")
                    else:
                        logger.warning(f"Skipping invalid item: expected dict but got {type(item)}")
                
                logger.info(f"Found {len(board_game_items)} board game items out of {len(parsed_response)} total items")
                response_data = prepare_data(response_list=board_game_items)
                logger.info(f"Successfully processed {len(response_data)} games from batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")
                return response_data
            
            elif response.status_code == 429:
                wait_time = retry_delay * (max_retries - attempt)
                logger.warning(f"Received status {response.status_code}, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
    
            else:
                response.raise_for_status()
                
    except httpx.ConnectTimeout:
        raise RuntimeError(f"Connection timeout when fetching batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")
    except httpx.ReadTimeout:
        raise RuntimeError(f"Read timeout when fetching batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching batch {batch_ids.split(',')[0]} to {batch_ids.split(',')[-1]}: {str(e)}")
    
async def continuous_fetch() -> List[Dict[str, Any]]:
    """
    Continuously fetch board game data in batches until a maximum number of consecutive failures is reached.
    """
    max_retries = settings.INGESTION['max_retries']
    retry_delay = settings.INGESTION['retry_delay']
    batch_size = settings.INGESTION['batch_size']
    async_requests = settings.INGESTION['async_requests']
    max_consecutive_failures = settings.INGESTION['max_consecutive_failures']
    endpoint = settings.INGESTION['endpoint']
    base_params = settings.INGESTION['base_params']

    logger.info(f"Starting continuous fetch with batch_size={batch_size}, async_requests={async_requests}")

    current_id = 0
    consecutive_failures = 0
    total_processed = 0

    data_dir = settings.PATHS['raw_data']
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / f'{execution_timekey}_bgg.parquet'

    async with httpx.AsyncClient() as client:
        logger.info(f"Created HTTP client for batch requests")
        while consecutive_failures < max_consecutive_failures:
            try:
                batch_starts = [current_id + (batch_size * i) + 1 for i in range(async_requests)]
                request_batches = [",".join(str(start + j) for j in range(batch_size)) for start in batch_starts]
                
                logger.info(f"Processing {async_requests} batches from ID {batch_starts[0]} to {batch_starts[-1] + batch_size - 1}")
                
                tasks = [fetch_bgg_data(
                    endpoint=endpoint, 
                    params=base_params, 
                    batch_ids=batch, 
                    max_retries=max_retries, 
                    retry_delay=retry_delay, 
                    async_client=client
                ) for batch in request_batches]
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_valid_results = 0

                valid_results = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch {i} failed with error: {result}")
                    elif result and isinstance(result, list):
                        batch_valid_results += len(result)
                        valid_results.extend(result)

                if batch_valid_results:
                    if os.path.exists(parquet_path):
                        existing_data = pl.read_parquet(parquet_path)
                        combined_df = pl.concat([existing_data, pl.DataFrame(valid_results)])
                        combined_df.write_parquet(parquet_path, compression='lz4')
                        logger.info(f"Appended {batch_valid_results} new results to existing Parquet file: {parquet_path}")
                    else:
                        pl.DataFrame(valid_results).write_parquet(parquet_path, compression='lz4')
                        logger.info(f"Created new Parquet file with {batch_valid_results} results: {parquet_path}")
                
                    total_processed += batch_valid_results
                    consecutive_failures = 0
                    logger.info(f"Batch successful: Total processed: {total_processed}")
                else:
                    consecutive_failures += 1
                    logger.warning(f"No valid results found in batch. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                
                current_id += batch_size * async_requests
                logger.info(f"Moving to next batch starting at ID {current_id + 1}")

            except Exception as e:
                await client.aclose()
                logger.error(f"Error in continuous fetch: {e}", exc_info=True)
                sys.exit(1)

    logger.info(f"Continuous fetch completed. Total games fetched: {total_processed}")
    return

if __name__ == "__main__":
    asyncio.run(continuous_fetch())
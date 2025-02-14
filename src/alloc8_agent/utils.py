import asyncio
import json
import os
import re
import ssl

import aiohttp
import certifi
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from subgrounds import Subgrounds

from alloc8_agent.constants import (
    CAMELOT_QUERY_MAPPING,
    VALID_CHAINS,
    RISK_PARAMS, STABLE_COINS_CAMELOT, BLUECHIP_COINS_CAMELOT, STABLE_COINS_AERODROME,
    BLUECHIP_COINS_AERODROME,
)

load_dotenv()
SUBGRAPH_API_KEY = os.getenv('SUBGRAPH_API_KEY')

ssl_context = ssl.create_default_context(cafile=certifi.where())


# def fetch_pools(pools_filters, limit=10):
#     """
#     Fetch liquidity pool data from blockchain subgraphs based on filters.
#
#     Args:
#         pools_filters (dict): Filters including chain, pool type, and TVL sorting
#         limit (int): Maximum number of pools to return per chain
#
#     Returns:
#         pd.DataFrame: Combined pool data from specified chains
#     """
#     try:
#         with Subgrounds() as sg:
#             chain = pools_filters.get('chain', "all")
#             order_by = "totalValueLockedUSD" if pools_filters.get("tvlUSD") == "high" else "low"
#             order_direction = "desc" if pools_filters.get(order_by) == "high" else "asc"
#             risk_level = pools_filters.get("risk", "low")
#             pool_type = pools_filters.get("pool_type", "stablecoin")
#
#             if chain == "arbitruim":
#                 STABLE_COINS = STABLE_COINS_CAMELOT
#                 BLUECHIP_COINS = BLUECHIP_COINS_CAMELOT
#                 tvl_threshold = 500000
#             elif chain == "base":
#                 STABLE_COINS = STABLE_COINS_AERODROME
#                 BLUECHIP_COINS = BLUECHIP_COINS_AERODROME
#                 tvl_threshold = 1000000
#             else:
#                 print("yhaan", chain)
#                 STABLE_COINS = STABLE_COINS_CAMELOT + STABLE_COINS_AERODROME
#                 BLUECHIP_COINS = BLUECHIP_COINS_CAMELOT + BLUECHIP_COINS_AERODROME
#                 tvl_threshold = 1000000
#
#             print(STABLE_COINS, "<-->", BLUECHIP_COINS)
#             print(f"DEBUG: Chain = {chain}")
#             print(f"DEBUG: Risk Level = {risk_level}")
#             print(f"DEBUG: STABLE_COINS = {STABLE_COINS}")
#             print(f"DEBUG: BLUECHIP_COINS = {BLUECHIP_COINS}")
#
#             # Determine Token 1 and Token 2 based on risk strategy
#             token1_filter = []
#             token2_filter = []
#
#             if risk_level == "low":
#                 token1_filter = STABLE_COINS
#                 token2_filter = STABLE_COINS  # Pairing stablecoins with stablecoins
#                 range_type = "wide"
#
#             elif risk_level == "medium":
#                 token1_filter = STABLE_COINS
#                 token2_filter = BLUECHIP_COINS  # Medium risk: Stablecoins + Blue-chip
#                 range_type = "medium"
#
#             elif risk_level == "high":
#                 token1_filter = BLUECHIP_COINS
#                 token2_filter = BLUECHIP_COINS  # High risk: Blue-chip with Blue-chip
#                 range_type = "narrow"
#
#                 # Debugging output
#             print(f"DEBUG: token1_filter = {token1_filter}")
#             print(f"DEBUG: token2_filter = {token2_filter}")
#
#             print(token2_filter)
#             where_conditions = {
#                 "totalValueLockedUSD_gt": tvl_threshold,
#                 "token0_": {"symbol_in": token1_filter},
#                 "token1_": {"symbol_in": token2_filter}
#             }
#             print(where_conditions)
#             # Determine which chains to fetch data from
#             if chain == 'all':
#                 print("here")
#                 chains_to_fetch = VALID_CHAINS.keys()
#             else:
#                 chains_to_fetch = [chain] if chain in VALID_CHAINS else []
#
#             # Fetch data from each chain and combine results
#             results = []
#             for ch in chains_to_fetch:
#                 subgraph_url = VALID_CHAINS[ch].format(api_key=SUBGRAPH_API_KEY)
#                 camelot = sg.load_subgraph(subgraph_url)
#
#                 pools = camelot.Query.pools(
#                     where=where_conditions,
#                     first=limit,
#                     orderBy=order_by,
#                     orderDirection=order_direction
#                 )
#
#                 df = sg.query_df(
#                     [
#                         pools.id,
#                         pools.totalValueLockedUSD,
#                         pools.txCount,
#                         pools.token0.id,
#                         pools.token0.symbol,
#                         pools.token1.id,
#                         pools.token1.symbol
#                     ],
#                     columns=[
#                         "pool_id",
#                         "tvl_usd",
#                         "txn_count",
#                         "token0_id",
#                         "token0_symbol",
#                         "token1_id",
#                         "token1_symbol",
#                     ]
#                 )
#                 print(df)
#                 df["chain"] = ch
#                 df["range"] = range_type
#                 df["risk"] = risk_level
#                 results.append(df)
#
#             final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
#             return final_df
#     except Exception as e:
#         print(f"ERROR: fetch_pools {e}")
#
def fetch_pools(pools_filters, limit=10):
    """
    Fetch liquidity pool data from blockchain subgraphs based on filters.

    Args:
        pools_filters (dict): Filters including chain, pool type, TVL sorting, and symbol
        limit (int): Maximum number of pools to return per chain

    Returns:
        pd.DataFrame: Combined pool data from specified chains
    """
    try:
        with Subgrounds() as sg:
            chain = pools_filters.get('chain', "all")
            order_by = "totalValueLockedUSD" if pools_filters.get("tvlUSD") == "high" else "low"
            order_direction = "desc" if pools_filters.get(order_by) == "high" else "asc"
            risk_level = pools_filters.get("risk", "low")
            pool_type = pools_filters.get("pool_type", "stablecoin")
            symbol_filter = pools_filters.get("symbol", "")

            if chain == "arbitrum":
                STABLE_COINS = STABLE_COINS_CAMELOT
                BLUECHIP_COINS = BLUECHIP_COINS_CAMELOT
                tvl_threshold = 500000
            elif chain == "base":
                STABLE_COINS = STABLE_COINS_AERODROME
                BLUECHIP_COINS = BLUECHIP_COINS_AERODROME
                tvl_threshold = 1000000
            else:
                STABLE_COINS = STABLE_COINS_CAMELOT + STABLE_COINS_AERODROME
                BLUECHIP_COINS = BLUECHIP_COINS_CAMELOT + BLUECHIP_COINS_AERODROME
                tvl_threshold = 1000000

            # Determine Token 1 and Token 2 based on filters
            if symbol_filter:
                # If a specific symbol is provided, prioritize it for token1
                token1_filter = [symbol_filter.upper()]
                # Determine token2 based on risk level
                if risk_level == "low":
                    token2_filter = STABLE_COINS
                    range_type = "wide"
                elif risk_level == "medium":
                    token2_filter = BLUECHIP_COINS
                    range_type = "medium"
                elif risk_level == "high":
                    token2_filter = BLUECHIP_COINS
                    range_type = "narrow"
                else:
                    token2_filter = STABLE_COINS + BLUECHIP_COINS
                    range_type = "wide"
            else:
                # If no specific symbol is provided, use pool_type and risk_level to determine filters
                if pool_type == "stablecoin":
                    token1_filter = STABLE_COINS
                elif pool_type == "bluechip":
                    token1_filter = BLUECHIP_COINS
                else:
                    token1_filter = STABLE_COINS + BLUECHIP_COINS

                if risk_level == "low":
                    token2_filter = STABLE_COINS
                    range_type = "wide"
                elif risk_level == "medium":
                    token2_filter = BLUECHIP_COINS
                    range_type = "medium"
                elif risk_level == "high":
                    token2_filter = BLUECHIP_COINS
                    range_type = "narrow"
                else:
                    token2_filter = STABLE_COINS + BLUECHIP_COINS
                    range_type = "wide"

            where_conditions = {
                "totalValueLockedUSD_gt": tvl_threshold,
                "token0_": {"symbol_in": token1_filter},
                "token1_": {"symbol_in": token2_filter}
            }
            # Determine which chains to fetch data from
            if chain == 'all':
                chains_to_fetch = VALID_CHAINS.keys()
            else:
                chains_to_fetch = [chain] if chain in VALID_CHAINS else []

            # Fetch data from each chain and combine results
            results = []
            for ch in chains_to_fetch:
                subgraph_url = VALID_CHAINS[ch].format(api_key=SUBGRAPH_API_KEY)
                camelot = sg.load_subgraph(subgraph_url)

                pools = camelot.Query.pools(
                    where=where_conditions,
                    first=limit,
                    orderBy=order_by,
                    orderDirection=order_direction
                )

                df = sg.query_df(
                    [
                        pools.id,
                        pools.totalValueLockedUSD,
                        pools.txCount,
                        pools.token0.id,
                        pools.token0.symbol,
                        pools.token1.id,
                        pools.token1.symbol
                    ],
                    columns=[
                        "pool_id",
                        "tvl_usd",
                        "txn_count",
                        "token0_id",
                        "token0_symbol",
                        "token1_id",
                        "token1_symbol",
                    ]
                )
                df["chain"] = ch
                df["range"] = range_type
                df["risk"] = risk_level
                results.append(df)

            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
            return final_df
    except Exception as e:
        print(f"ERROR: fetch_pools {e}")


# ----------------------------- Price Analysis Metrics -----------------------------

def calculate_volatility(price_data):
    """
    Calculate annualized volatility from historical price data.

    Args:
        price_data (dict): Pool's daily price data from subgraph

    Returns:
        dict: Volatility percentages for both tokens in the pool
    """
    pool_day_data = price_data.get('data', {}).get('pool', {}).get('poolDayData', [])

    if not pool_day_data:
        return {"token0_volatility": 0.0, "token1_volatility": 0.0}

    df = pd.DataFrame(pool_day_data)

    # Ensure price columns exist and convert to float
    if 'token0Price' not in df or 'token1Price' not in df:
        return {"token0_volatility": 0.0, "token1_volatility": 0.0}

    df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
    df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')

    # Drop NaN values before calculating percentage change
    vol_token0 = df['token0Price'].pct_change().dropna().std() * np.sqrt(365)
    vol_token1 = df['token1Price'].pct_change().dropna().std() * np.sqrt(365)

    # Handle NaN cases
    vol_token0 = round(vol_token0, 4) if pd.notna(vol_token0) else 0.0
    vol_token1 = round(vol_token1, 4) if pd.notna(vol_token1) else 0.0

    return {
        "token0_volatility": vol_token0,
        "token1_volatility": vol_token1
    }


def calculate_liquidity_concentration(liquidity_data):
    """
    Calculate liquidity concentration metrics (HHI and Gini Coefficient).

    Args:
        liquidity_data (dict): Pool's liquidity distribution data

    Returns:
        dict: HHI and Gini metrics
    """

    ticks = liquidity_data.get('data', {}).get('pool', {}).get('ticks', [])

    liquidity_values = [abs(int(tick.get('liquidityNet', 0))) for tick in ticks if
                        int(tick.get('liquidityNet', 0)) != 0]

    total_liquidity = sum(liquidity_values)
    if total_liquidity == 0:
        return {"HHI": 0, "Gini Coefficient": 0}

    # Normalize liquidity values
    normalized_liquidity = [liq / total_liquidity for liq in liquidity_values]

    # Calculate Herfindahl-Hirschman Index (HHI)
    hhi = sum(x ** 2 for x in normalized_liquidity)

    # Calculate Gini Coefficient
    sorted_liq = sorted(normalized_liquidity)
    n = len(sorted_liq)

    if n == 0:
        return {"HHI": round(hhi, 4), "Gini Coefficient": 0}

    gini = (2 * sum((i + 1) * liq for i, liq in enumerate(sorted_liq)) / (n * sum(sorted_liq))) - (n + 1) / n

    return {
        "HHI": round(hhi, 4),
        "Gini Coefficient": round(gini, 4)
    }


def calculate_volume_fee_analysis(volume_data):
    """
    Analyze trading volume and fee metrics.

    Args:
        volume_data (dict): Pool's volume and fee data

    Returns:
        dict: Aggregated volume, fees, and TVL metrics
    """
    pool_data = volume_data.get('data', {}).get('pool', {})
    pool_day_data = pool_data.get('poolDayData', [])

    if not pool_day_data:
        return {
            "total_vol_usd": 0.0,
            "total_fees_usd": 0.0,
            "avg_fee_rate": 0.0,
            "avg_tvl_usd": 0.0
        }

    df = pd.DataFrame(pool_day_data)

    if 'feesUSD' not in df or 'volumeUSD' not in df:
        return {
            "total_vol_usd": 0.0,
            "total_fees_usd": 0.0,
            "avg_fee_rate": 0.0,
            "avg_tvl_usd": 0.0
        }

    # Convert necessary fields to float
    df['feesUSD'] = pd.to_numeric(df['feesUSD'], errors='coerce').fillna(0.0)
    df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce').fillna(0.0)

    # Compute required metrics
    total_volume = df['volumeUSD'].sum()
    total_fees = df['feesUSD'].sum()
    avg_fee_rate = (total_fees / total_volume) if total_volume != 0 else 0

    total_value_locked = pool_data.get('totalValueLockedUSD', 0.0)
    average_tvl = float(total_value_locked) if total_value_locked else 0.0

    return {
        "total_vol_usd": round(total_volume, 2),
        "total_fees_usd": round(total_fees, 2),
        "avg_fee_rate": round(avg_fee_rate, 6),
        "avg_tvl_usd": round(average_tvl, 2)
    }


def calculate_apr(apr_data, tvl, average_tvl):
    """
    Calculate Annual Percentage Return metrics.

    Args:
        apr_data (dict): Fee data from different time ranges
        tvl (float): Current Total Value Locked
        average_tvl (float): Historical average TVL

    Returns:
        dict: Daily and average APR percentages
    """
    daily_fees = float(apr_data.get('data', {}).get('pool', {}).get('daily', [{}])[0].get('feesUSD', 0))
    monthly_fees = sum(
        float(day.get('feesUSD', 0)) for day in apr_data.get('data', {}).get('pool', {}).get('monthly', []))

    print(f"Daily Fees: {daily_fees}, Monthly Fees: {monthly_fees}, TVL: {average_tvl}")

    daily_apr = (daily_fees * 365 * 100) / float(tvl or 1)
    average_apr = (monthly_fees * 12 * 100) / float(average_tvl or 1)

    return {
        "daily_apr": round(daily_apr, 4),
        "average_apr": round(average_apr, 4)
    }


def calculate_pool_price(token0_price, token1_price, total_locked_token0, total_locked_token1):
    """
    Calculate weighted average pool price.

    Args:
        token0_price (float): Price of first token
        token1_price (float): Price of second token
        total_locked_token0 (float): TVL in token0
        total_locked_token1 (float): TVL in token1

    Returns:
        float: Weighted average price or None if invalid input
    """
    try:
        # Convert all values to float
        token0_price = float(token0_price) if token0_price is not None else None
        token1_price = float(token1_price) if token1_price is not None else None
        total_locked_token0 = float(total_locked_token0) if total_locked_token0 is not None else None
        total_locked_token1 = float(total_locked_token1) if total_locked_token1 is not None else None

        if None in [token0_price, token1_price, total_locked_token0, total_locked_token1]:
            return None

        if total_locked_token0 + total_locked_token1 == 0:
            return None

        pool_price = ((total_locked_token0 * token0_price) + (total_locked_token1 * token1_price)
                      ) / (total_locked_token0 + total_locked_token1)

        return pool_price

    except ValueError:
        print("Error: Non-numeric value encountered")
        return None


async def analyze_pool(session, pool_id, semaphore, chain, risk):
    """
    Async wrapper for pool analysis pipeline.

    Args:
        session (aiohttp.ClientSession): HTTP session
        pool_id (str): Pool ID to analyze
        semaphore (asyncio.Semaphore): Concurrency limiter
        chain (str): Blockchain network identifier

    Returns:
        dict: Aggregated analysis results for the pool
    """
    pool_id = pool_id.lower()
    tasks = [
        fetch_data_async(session, pool_id, "price", semaphore, chain),
        fetch_data_async(session, pool_id, "vol", semaphore, chain),
        fetch_data_async(session, pool_id, "liq", semaphore, chain),
        fetch_data_async(session, pool_id, "apr", semaphore, chain),
        fetch_data_async(session, pool_id, "ohlc", semaphore, chain)
    ]
    price_data, vol_data, liq_data, apr_data, ohlc_data = await asyncio.gather(*tasks)

    try:

        liq_data = liq_data if isinstance(liq_data, dict) else {}
        price_metadata = price_data.get('data', {}).get('pool', {})
        tvl = float(liq_data.get('data', {}).get('pool', {}).get('totalValueLockedUSD', 1))

        volatility_metrics = calculate_volatility(price_data)
        liquidity_metrics = calculate_liquidity_concentration(liq_data)
        volume_fee_metrics = calculate_volume_fee_analysis(vol_data)
        apr_metrics = calculate_apr(apr_data=apr_data, tvl=tvl, average_tvl=volume_fee_metrics.get('avg_tvl_usd', 0))
        current_price = calculate_pool_price(token0_price=price_metadata.get('token0Price'),
                                             token1_price=price_metadata.get('token1Price'),
                                             total_locked_token0=price_metadata.get(
                                                 'totalValueLockedToken0'),
                                             total_locked_token1=price_metadata.get(
                                                 'totalValueLockedToken1'))
        ranges = get_ranges(ohlc_data=ohlc_data, risk=risk, current_price=current_price)

        result = {"pool_id": pool_id, "current_price": current_price, **volatility_metrics, **liquidity_metrics,
                  **volume_fee_metrics, **apr_metrics, **ranges}

        return result
    except Exception as e:
        print(f"Error analyze_pool: {e}")
        return f"No Data found for this pool: {pool_id}to analyze"


async def fetch_data_async(session, pool_id, data_type, semaphore, chain):
    """
      Async fetch GraphQL data for a pool.

      Args:
          session: aiohttp session
          pool_id: Target pool ID
          data_type: One of 'price', 'vol', 'liq', 'apr'
          semaphore: Concurrency control
          chain: Blockchain identifier

      Returns:
          dict: JSON response from subgraph
      """
    async with semaphore:
        query = CAMELOT_QUERY_MAPPING[data_type].format(pool_id=pool_id)
        url = VALID_CHAINS[chain].format(api_key=SUBGRAPH_API_KEY)
        try:
            async with session.post(url, json={"query": query}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching {data_type} for {pool_id}: {response.status}")
                    return {}
        except Exception as e:
            print(f"Exception fetching {data_type} for {pool_id}: {e}")
            return {}


async def analyze_pools_async(pool_ids, chains, risk):
    """
       Main async analysis entry point.

       Args:
           pool_ids (list): List of pool IDs to analyze
           chains (list): Corresponding chain identifiers

       Returns:
           pd.DataFrame: Analysis results for all pools
       """
    semaphore = asyncio.Semaphore(10)

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [analyze_pool(session, pool_id, semaphore, chain, risk) for pool_id, chain in zip(pool_ids, chains)]
        results = await asyncio.gather(*tasks)

    return pd.DataFrame([r for r in results if r])


# ----------------------------- Utility Functions -----------------------------
def parse_json_string(input_string: str) -> dict:
    """
      Clean and parse JSON from potentially malformed strings.

      Args:
          input_string (str): String containing JSON data

      Returns:
          dict: Parsed JSON data

      Raises:
          ValueError: If invalid JSON after cleaning
      """
    # Remove backticks and words 'json' and 'python'
    cleaned_string = input_string.replace('`', '').replace('json', '').replace('python', '').strip()

    # Remove Python-style comments (lines that start with #)
    cleaned_string = re.sub(r"#.*", "", cleaned_string)

    # Replace tuple-like values with JSON-compatible lists
    cleaned_string = re.sub(r"\(\s*([\d\.\-e]+)\s*,\s*([\d\.\-e]+)\s*\)", r"[\1, \2]", cleaned_string)

    # Convert to dictionary
    print(cleaned_string)
    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def analyze_wallet(wallet_address):
    """Placeholder for future wallet analysis functionality."""
    return "Oops !! Something Went Wrong Retry in a while ... ):"


def get_market_regime(row):
    adx_threshold = 20
    bb_squeeze_threshold = 0.1 * row['close']  # 10% of price

    # Check Bollinger Band squeeze
    bb_width = row['bbu_20_2.0'] - row['bbl_20_2.0']
    is_squeeze = bb_width < bb_squeeze_threshold

    if row['adx_14'] < adx_threshold and is_squeeze and 30 < row['rsi_14'] < 70:
        return "Range-Bound"
    elif row['adx_14'] > adx_threshold:
        return "Trending"
    else:
        return "Uncertain"


def calculate_indicators(df):
    # Calculate Bollinger Bands (20-day, 2σ)
    df.ta.bbands(length=20, std=2, append=True)

    # Calculate ADX (14-day)
    df.ta.adx(length=14, append=True)

    # Calculate RSI (14-day)
    df.ta.rsi(length=14, append=True)

    # Clean column names
    df.columns = [col.lower() for col in df.columns]

    # Drop rows with NaN values (these are the initial rows without enough data)
    df = df.dropna().reset_index(drop=True)

    return df


def calculate_lp_ranges(df, risk_profile="medium"):
    params = RISK_PARAMS[risk_profile]
    latest = df.iloc[-1]  # Latest data point
    # latest = latest.apply(pd.to_numeric, errors='coerce')  # Convert to float

    # Base Bollinger Bands
    bb_upper = latest['bbu_20_2.0']
    bb_lower = latest['bbl_20_2.0']
    price = latest['close']

    # Volatility-adjusted buffer
    bb_width = bb_upper - bb_lower
    buffer = bb_width * params["volatility_multiplier"]

    # RSI directional bias
    if latest['rsi_14'] > (70 - params["rsi_buffer"]):
        rsi_bias = -0.02  # Overbought: lean lower
    elif latest['rsi_14'] < (30 + params["rsi_buffer"]):
        rsi_bias = 0.02  # Oversold: lean higher
    else:
        rsi_bias = 0

    # ADX-based regime adjustment
    if latest['adx_14'] > params["adx_trend_threshold"]:
        # Trending market: asymmetric range
        trend_strength = latest['adx_14'] / 100
        upper = price * (1 + (0.15 * trend_strength) + rsi_bias)
        lower = price * (1 - (0.10 * trend_strength) + rsi_bias)
    else:
        # Range-bound market: symmetric range
        upper = price + buffer + (price * rsi_bias)
        lower = price - buffer + (price * rsi_bias)

    # Use more decimal places since values are very small
    return round(lower, 6), round(upper, 6)


def get_ranges(ohlc_data, risk, current_price):

    data = ohlc_data['data']['poolDayDatas']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.sort_values('date').reset_index(drop=True)
    df[['high', 'low', 'close', 'volumeUSD']] = df[['high', 'low', 'close', 'volumeUSD']].astype(float)

    df = calculate_indicators(df)
    recommendations = {}
    lower, upper = calculate_lp_ranges(df, risk)
    lower_pct = ((lower - current_price) / current_price) * 100
    upper_pct = ((upper - current_price) / current_price) * 100
    recommendations[risk] = {
        "range": (lower, upper),
        "upper_limit_pct": f"{upper_pct:.3f}%",
        "lower_limit_pct": f"{lower_pct:.3f}%",
        "fee_rank": "🟢 Low" if risk == "low" else "🟡 Medium" if risk == "medium" else "🔴 High"
    }
    return {
        "pool_range": (lower, upper),
        "upper_limit_pct": f"{upper_pct:.3f}%",
        "lower_limit_pct": f"{lower_pct:.3f}%",
        "fee_rank": "🟢 Low" if risk == "low" else "🟡 Medium" if risk == "medium" else "🔴 High"
    }

STRATEGY_TYPE_WITH_RISK = {
    "Narrow": "low",
    "Wide": "mid",
    "Cowcentrated": "high",
    "SLP Wide": "mid",
    "SLP Bear": "high",
    "SLP Narrow": "high",
    "SLP Bull": "mid",
    "ALO": "mid",
    "Pegged": "low",
    "CRSV": "mid",
    "Stable": "low",
    "HLCS": "mid",
    "GSSS": "mid",
    "EES": "mid",
    "SLP Pegged": "low",
    "Smart LP Legacy": "mid",
    "MVCS": "high",
    "SSSV": "high",
    "MSS": "high",
    "HLCSV": "mid"
}

CAMELOT_VAULT_DATA_API = "https://api.camelot.exchange/vaults?chainId=42161"

VOLATILITY_QUERY = """
{{
  pool(id: "{pool_id}") {{
    token0Price
    token1Price
    tick
    totalValueLockedToken0
    totalValueLockedToken1
    poolDayData(first: 30, orderBy: date, orderDirection: desc) {{
      token0Price
      token1Price
      date
    }}
  }}
}}
"""

LIQUIDITY_CONCENTRATION_QUERY = """
{{
  pool(id: "{pool_id}") {{
    liquidity
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
    liquidityProviderCount
    # Liquidity concentration
    ticks(first: 1000, orderBy: tickIdx) {{
      tickIdx
      liquidityNet
    }}
  }}
}}

"""

VOLUME_FEE_QUERY = """
{{
  pool(id: "{pool_id}") {{
    volumeUSD
    feesUSD
    txCount
    totalValueLockedUSD
    # Historical volume analysis
    poolDayData(first: 30, orderBy: date, orderDirection: desc) {{
      volumeUSD
      feesUSD
    }}
  }}
}}

"""

APR_HISTORIC_QUERY = """
{{
  pool(id: "{pool_id}") {{
    feesUSD
    totalValueLockedUSD
    # 24h data
    daily: poolDayData(first: 1, orderBy: date, orderDirection: desc) {{
      feesUSD
      date
    }}
    # 30-day data
    monthly: poolDayData(first: 30, orderBy: date, orderDirection: desc) {{
      feesUSD
      date
    }}
  }}
}}
"""

OHLC_HISTORIC_QUERY = """
{{
  poolDayDatas(
    first: 30
    where: {{ pool: "{pool_id}" }}
    orderBy: date
    orderDirection: desc
  ) {{
    date
    high
    low
    close
    volumeUSD
  }}
}}
"""

CAMELOT_QUERY_MAPPING = {"vol": VOLUME_FEE_QUERY, "liq": LIQUIDITY_CONCENTRATION_QUERY, "price": VOLATILITY_QUERY,
                         "apr": APR_HISTORIC_QUERY, "ohlc": OHLC_HISTORIC_QUERY}

STABLE_COINS_CAMELOT = ["USDC", "USDT", "DAI", "USDe"]
BLUECHIP_COINS_CAMELOT = ["ETH", "WBTC", "SOL", "weETH", "wstETH", "ezETH"]

STABLE_COINS_AERODROME = ["USDC", "USDT", "DOLA", "eUSD", "USD+", "USDbC", "DAI"]
BLUECHIP_COINS_AERODROME = ["WETH", "SuperOETHb", "wstETH", "msETH", "cbETH", "ezETH", "weETH"]

CAMELOT_SUBGRAPH_URL = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/3utanEBA9nqMjPnuQP1vMCCys6enSM3EawBpKTVwnUw2"
AERODOME_SUBGRAPH_URL = "https://gateway.thegraph.com/api/{api_key}/subgraphs/id/GENunSHWLBXm59mBSgPzQ8metBEp9YDfdqwFr91Av1UM"

VALID_CHAINS = {
    "base": AERODOME_SUBGRAPH_URL,
    "arbitruim": CAMELOT_SUBGRAPH_URL
}

RISK_PARAMS = {
    "low": {
        "volatility_multiplier": 2.0,  # Wider ranges
        "rsi_buffer": 15,             # Avoid RSI extremes
        "adx_trend_threshold": 25     # Conservative trend detection
    },
    "medium": {
        "volatility_multiplier": 1.5,
        "rsi_buffer": 10,
        "adx_trend_threshold": 20
    },
    "high": {
        "volatility_multiplier": 1.0,  # Tighter ranges
        "rsi_buffer": 5,
        "adx_trend_threshold": 15     # Aggressive trend reaction
    }
}

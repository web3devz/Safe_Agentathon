import numpy as np
import requests
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from typing import Dict, List, Optional


class CamelotDataFetcher:
    def __init__(self, graph_api_key: str):
        self.subgraph_url = f"https://gateway.thegraph.com/api/{graph_api_key}/subgraphs/id/3utanEBA9nqMjPnuQP1vMCCys6enSM3EawBpKTVwnUw2"
        self.vaults_api = "https://api.camelot.exchange/vaults?chainId=42161"

        # Initialize GraphQL client
        transport = RequestsHTTPTransport(url=self.subgraph_url)
        self.client = Client(transport=transport, fetch_schema_from_transport=True)

    def _run_query(self, query: str, params: Dict = None) -> Dict:
        """Execute GraphQL query"""
        try:
            return self.client.execute(gql(query), variable_values=params)
        except Exception as e:
            print(f"Query failed: {e}")
            return {}

    def get_pool_data(self, pool_id: str) -> Dict:
        """Get comprehensive pool data for risk analysis"""
        query = gql("""
        query GetPoolData($poolId: ID!) {
            pool(id: $poolId) {
                liquidity
                totalValueLockedUSD
                volumeUSD
                feesUSD
                token0Price
                token1Price
                liquidityProviderCount
                poolDayData(first: 30, orderBy: date, orderDirection: desc) {
                    date
                    volumeUSD
                    feesUSD
                    token0Price
                    token1Price
                    totalValueLockedUSD
                }
                ticks(first: 1000, orderBy: tickIdx) {
                    tickIdx
                    liquidityNet
                }
            }
        }
        """)
        return self._run_query(query, {"poolId": pool_id})

    def get_vaults(self) -> List[Dict]:
        """Get Camelot vault strategies"""
        response = requests.get(self.vaults_api)
        return response.json() if response.status_code == 200 else []

    def calculate_risk_metrics(self, pool_data: Dict) -> Dict:
        """Calculate risk metrics from raw pool data"""
        # Calculate APR
        daily_fees = float(pool_data['pool']['feesUSD'])
        tvl = float(pool_data['pool']['totalValueLockedUSD'])
        apr = (daily_fees * 365 * 100) / tvl if tvl > 0 else 0

        # Calculate volatility
        prices = [float(d['token0Price']) for d in pool_data['pool']['poolDayData']]
        volatility = (max(prices) - min(prices)) / np.mean(prices) if prices else 0

        # Calculate liquidity concentration
        ticks = pool_data['pool']['ticks']
        total_liquidity = sum(abs(float(t['liquidityNet'])) for t in ticks)
        top_ticks = sorted(ticks, key=lambda x: abs(float(x['liquidityNet'])), reverse=True)[:5]
        concentration = sum(abs(float(t['liquidityNet'])) for t in top_ticks) / total_liquidity

        return {
            "apr": apr,
            "volatility": volatility,
            "liquidity_concentration": concentration,
            "volume_tvl_ratio": float(pool_data['pool']['volumeUSD']) / tvl if tvl > 0 else 0,
            "provider_count": int(pool_data['pool']['liquidityProviderCount'])
        }

    def get_optimal_strategies(self, risk_profile: str) -> List[Dict]:
        """Get strategies matching risk profile"""
        strategies = self.get_vaults()
        risk_map = {
            "low": {"min_apr": 5, "max_volatility": 0.2},
            "medium": {"min_apr": 10, "max_volatility": 0.5},
            "high": {"min_apr": 15, "max_volatility": 1.0}
        }

        return [
            s for s in strategies
            if s['apr'] >= risk_map[risk_profile]['min_apr'] and
               s['volatility'] <= risk_map[risk_profile]['max_volatility']
        ]

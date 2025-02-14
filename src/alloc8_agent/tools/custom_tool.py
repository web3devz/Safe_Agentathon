import asyncio
import json

from crewai.tools import BaseTool
from typing import Dict
from pydantic import BaseModel, Field

from alloc8_agent.utils import fetch_pools, analyze_pools_async


class GetPoolDataInput(BaseModel):
    pool_filters: Dict = Field(..., description="Dictionary containing pool filters: pool_type, tvlUSD, chain")


class GetPoolData(BaseTool):
    name: str = "GetPoolData "
    description: str = (
        "Get pool data with KPIs and analytical values using dictionary filters {pool_type, tvlUSD, chain}"
    )

    def _run(self, pool_filters: dict) -> str:
        """
         Get pool data as per filters.
        """
        try:
            if not isinstance(pool_filters, dict):
                raise ValueError("Input must be a dictionary with keys: pool_type, tvlUSD, chain")

            validated_filters = {
                "pool_type": pool_filters.get("pool_type", "all"),
                "tvlUSD": pool_filters.get("tvlUSD", "high"),
                "chain": pool_filters.get("chain", "all"),
                "risk": pool_filters.get("risk", "low"),
                "symbol": pool_filters.get("symbol", "")
            }

            pool_df = fetch_pools(pools_filters=validated_filters, limit=5)
            if not pool_df.empty:
                pool_ids = pool_df["pool_id"].tolist()
                chains = pool_df["chain"].tolist()


                result_df = asyncio.run(analyze_pools_async(pool_ids=pool_ids, chains=chains,risk=validated_filters.get('risk')))

                merged_df = pool_df.merge(result_df, on="pool_id", how="left")

                return merged_df.to_json(orient='records')
            else:
                return json.dumps([])

        except Exception as e:
            print(f"Error :{e}")


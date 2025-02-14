#!/usr/bin/env python

from itertools import cycle
import warnings

warnings.simplefilter("ignore", ResourceWarning)

from alloc8_agent.crew import Alloc8Agent
from alloc8_agent.utils import parse_json_string, analyze_wallet


def run(user_input: str):
    try:
        alloc8_crew = Alloc8Agent().crew()
        result = alloc8_crew.kickoff(inputs={"user_input": user_input})
        final_out = result.raw
        if final_out:
            return parse_json_string(input_string=final_out)
        return []
    except Exception as e:
        return [{"error": f"Error processing request: {str(e)}"}]


import streamlit as st
import time

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .appview-container {
            background-color: #121212;
        }
        .stTextInput input {
            background-color: #333;
            color: white;
            border: 1px solid #555;
        }
        .scrollable-response {
            max-height: 70vh;
            overflow-y: auto;
            padding-bottom: 80px;
        }
        .pool-card {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 2px 2px 10px rgba(255, 255, 255, 0.1);
            border-left: 5px solid #6c63ff;
        }
        .sidebar-item {
            padding: 10px;
            cursor: pointer;
            border-radius: 5px;
            text-align: center;
            font-size: 16px;
            margin-bottom: 5px;
        }
        .sidebar-item:hover {
            background-color: #444;
        }
        .selected {
            background-color: #6c63ff !important;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True
)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "response" not in st.session_state:
    st.session_state["response"] = None

st.sidebar.title("🐱 Meowfi")

st.sidebar.header("🔍 Analysis Options")
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "LP Pool Analysis"

if st.sidebar.button("LP Pool Analysis", key="lp_pool"):
    st.session_state.selected_tab = "LP Pool Analysis"

st.sidebar.markdown("---")  # Adds a horizontal line for separation
st.sidebar.markdown(
    "[🐱 Meowfi on GitHub](https://github.com/yourdevkalki/Meowfi_agent.git) 🔗",
    unsafe_allow_html=True
)

if st.session_state.selected_tab == "LP Pool Analysis":
    st.title("🐱 Meowfi Liquidity Pools Explorer (Beta)")

    st.subheader("🔍 Discover Optimal Liquidity Pools on Arbitrum & Base")
    st.markdown('<div class="scrollable-response">', unsafe_allow_html=True)

    st.markdown("### 💡 Try These Queries:")
    st.markdown("- **I want to maximize my stablecoin yield with a high-risk strategy.**")
    st.markdown("- **How can I earn a stable and consistent yield on my USDC with minimal risk?**")
    st.markdown("- **I want to leverage my stablecoins moderately to boost returns without taking on too much risk.**")

    st.markdown("🚀 Currently supporting **Arbitrum** & **Base** networks.")

    if prompt := st.chat_input("Plan your strategy:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        stages = [
            ("🔄 Analyzing query...", 20),
            ("📊 Gathering liquidity pool data...", 40),
            ("📉 Calculating optimal strategies...", 60),
            ("⏳ Fine-tuning APR projections...", 80),
            ("📈 Finalizing optimal strategy...", 90)
        ]

        constant_message = st.empty()
        constant_message.text("⏳ Hang tight! Meowfi is planning your strategy... Might take 20-30 sec. Buckle up! 🚀🐱")

        for message, percent in stages:
            status_placeholder.text(message)
            progress_bar.progress(percent)
            time.sleep(1)

        response = run(prompt)

        # Clean up the progress bar and status text
        constant_message.empty()
        progress_bar.empty()
        status_placeholder.empty()

        # Save and display the response
        st.session_state["response"] = response
        st.session_state["messages"].append({"role": "assistant", "content": response})

        with st.chat_message("assistant", avatar="🐱"):
            if isinstance(response, list):
                for pool in response:
                    pool_details = pool.get("pool_details")
                    strategy = pool.get("lp_strategy")
                    tags = pool.get("tags", [])

                    tags_html = ""
                    if tags:
                        tags_html = '<div style="margin-top: 10px;">' + "".join(
                            f'<span style="display: inline-block; background-color: #007bff; color: white; padding: 5px 10px; border-radius: 15px; margin-right: 5px; font-size: 12px;">{tag}</span>'
                            for tag in tags
                        ) + "</div>"
                    token1_symbol = pool_details.get('token0_symbol', "")
                    formatted_fee_range = [f"{token1_symbol} {value}" for value in strategy['pool_range']]
                    st.markdown(
                        f"""
                        <div class="pool-card">
                            <h4>{pool_details['pool_name']} ({pool_details['pool_address']})</h4>
                            <p><strong>Current Daily APY:</strong> {pool_details['daily_apy']:.2f}%</p>
                            <p><strong>Total Liquidity:</strong> ${pool_details['total_liquidity']:,}</p>
                            <p><strong>Current Pool Price:</strong> {token1_symbol}{pool_details['current_price']}</p>
                            <p><strong>Optimal Liquidity Range %:</strong> [-{strategy['lower_limit_pct']},+{strategy['upper_limit_pct']} ]</p>
                            <p><strong>Volatility-Adjusted Range:</strong> [{', '.join(formatted_fee_range)}]</p>
                            <p><strong>Daily APR:</strong> {pool['projected_apr']['average_apr']:,}%</p>
                            <p><strong>Recommended Leverage:</strong> {pool['recommended_leverage']}</p>
                             {tags_html}
                        </div>
                        """, unsafe_allow_html=True
                    )
                    deposit_link = f"https://meowfi-alloc8.vercel.app/?id={pool_details['pool_address']}&upperRange={strategy['pool_range'][0]}&lowerRange={strategy['pool_range'][1]}"
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        # if st.button("💰 Deposit Here", key=f"deposit_{pool_details['pool_address']}"):
                        #     st.markdown(f"[Click to Deposit]({deposit_link})", unsafe_allow_html=True)
                        st.link_button("💰 Deposit Here", deposit_link)
                    with col2:
                        st.button("📈 Quant Analysis", key=f"quant_{pool_details['pool_address']}", disabled=True)

                    with col3:
                        st.button("❌ Withdraw", key=f"withdraw_{pool_details['pool_address']}",
                                  disabled=True)

                    with col4:
                        st.button("⚖️ Rebalance", key=f"rebalance_{pool_details['pool_address']}",
                                  disabled=True)
                    st.markdown(
                        f"""
                        <h4>🔹 Notes:</h4>
                        <ul>
                            <li><strong>Recalculate volatility daily</strong> using a 30-day rolling window.</li>
                            <li><strong>Threshold-Based Rebalancing:</strong>
                                <ul>
                                    <li>If price exits current range → Immediate adjustment.</li>
                                    <li>If volatility changes <strong>>15%</strong> from last calculation → Proactive adjustment.</li>
                                </ul>
                            </li>
                        </ul>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("---")
            else:
                st.write(response)

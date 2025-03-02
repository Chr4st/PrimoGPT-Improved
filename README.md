# Credits: https://github.com/ivebotunac/PrimoGPT 

## Overview

PrimoGPT is a modular trading system developed as part of doctoral research titled "An Automated Stock Market Trading System Based on Deep Reinforcement Learning". The system combines Natural Language Processing (NLP) and Deep Reinforcement Learning (DRL) to support investment decisions, aiming to minimize risk and potential losses through deep financial market analysis.

I am optimizing it further to the best of my abilities.

For further information, go to the original Repository: https://github.com/ivebotunac/PrimoGPT

## Improvements

1. The updated train.py script increases market adaptability by using Hidden Markov Models (HMM) for market regime detection and FinBERT for sentiment analysis, allowing the RL agent to adjust strategies dynamically. Additionally, a Sharpe Ratio-based reward function improves risk-adjusted returns, making the trading model more stable and profitable.
2. I added slippage and transaction cost simulation to test.py, making trade execution more realistic by adjusting prices with ±0.1% slippage and applying a 0.05% transaction cost per trade. This increases backtesting accuracy and ensures that PrimoRL’s performance metrics reflect real-world trading conditions.

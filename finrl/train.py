from __future__ import annotations
from finrl.config import ERL_PARAMS, INDICATORS, RLlib_PARAMS, SAC_PARAMS, TRAIN_END_DATE, TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from transformers import pipeline
from hmmlearn.hmm import GaussianHMM
import numpy as np

# NLP Sentiment Analysis using FinBERT
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def extract_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

# Market Regime Detection using HMM
def detect_market_regime(data):
    log_returns = np.log(data['close'] / data['close'].shift(1)).dropna()
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
    hmm_model.fit(log_returns.values.reshape(-1, 1))
    regimes = hmm_model.predict(log_returns.values.reshape(-1, 1))
    data['market_regime'] = regimes
    return data

# Custom Reward Function using Sharpe Ratio
def custom_reward(env, action, portfolio_value, previous_value, volatility):
    if previous_value == 0:
        return 0
    daily_return = (portfolio_value - previous_value) / previous_value
    sharpe_adjusted = daily_return / (volatility + 1e-6)
    return sharpe_adjusted

def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # Download and preprocess data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    data = detect_market_regime(data)  # Add Market Regime Detection
    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)
    
    cwd = kwargs.get("cwd", "./" + str(model_name))
    
    if drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
        
        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

if __name__ == "__main__":
    env = StockTradingEnv

    kwargs = {}
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        agent_params=ERL_PARAMS,
        total_timesteps=1e6,
        kwargs=kwargs,
    )

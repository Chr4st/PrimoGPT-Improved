from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)

class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")
                self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()
        else:
            raise ValueError("Data source input is NOT supported yet.")

        # Initialize variables
        self.tech_indicator_list = tech_indicator
        self.vix = vix

    def download_data(self, ticker_list, start_date, end_date, time_interval) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        df = df.dropna().reset_index(drop=True)  # Ensure no NaN values remain
        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)
        
        # Normalize technical indicators for better RL learning
        for indicator in tech_indicator_list:
            df[indicator] = (df[indicator] - df[indicator].mean()) / df[indicator].std()
        
        return df

    def add_macro_indicators(self, df) -> pd.DataFrame:
        # Example: Add macroeconomic indicators like interest rates, inflation, GDP growth
        macro_data = self.processor.get_macro_data()
        df = df.merge(macro_data, on="date", how="left")
        df = df.fillna(method='ffill')  # Forward-fill missing macro data
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)
        return df

    def add_time_decay_weighted_sentiment(self, df) -> pd.DataFrame:
        # Apply exponential decay to sentiment scores
        if 'sentiment_score' in df.columns:
            df['sentiment_score'] = df['sentiment_score'].ewm(span=10, adjust=False).mean()
        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        
        # Fill NaN and Inf values with 0 for technical indicators
        tech_array = np.nan_to_num(tech_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return price_array, tech_array, turbulence_array

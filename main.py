import numpy as np
import pandas as pd
# import streamlit as st

# D:\Python\Python38\Scripts\streamlit run main.py
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def date_window(start_time, end_time, data):
    data = data[(data['timestamp'] > start_time) & (data['timestamp'] < end_time)]
    return data


def cumulative_winners_losers(n_cusips, data, metric='spread_pnl'):
    data = data.sort_values(by=metric)

    winners = data[-n_cusips:].iloc[::-1]
    losers = data[:n_cusips]

    return winners, losers


def cumulative_change_window_gains_losses(n_cusips, data):
    data = data.sort_values(by='timestamp')
    first = data[['cusip', 'spread_pnl']].drop_duplicates('cusip')
    first.columns = ['cusip', 'beginning_spread_pnl']
    data = pd.merge(data, first, on='cusip')

    data['difference_since_beginning'] = data['spread_pnl'] - data['beginning_spread_pnl']
    data = data[['difference_since_beginning', 'cusip']].drop_duplicates('cusip', keep='last')

    return cumulative_winners_losers(n_cusips, data, metric='difference_since_beginning')


def deviation_bands(n_deviations, n_observations, data):
    deviation = data['spread_pnl'].rolling(n_observations).std() * n_deviations
    print(deviation)


def cumulative_pnl(data):
    cumulative = np.cumsum(data.groupby('timestamp').sum()['spread_pnl'])
    return cumulative


def filter_by_category(filter_column, desired_value, data):
    data = data[data[filter_column] == desired_value]
    return data


if __name__ == "__main__":

    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # print(date_window('2021-07-28 20:55:00', '2021-07-28 21:55:00', df))
    # print(winners_losers(5, df))
    # print(deviation_bands(1, 5, df))
    print(cumulative_change_window_gains_losses(5, df)[0])
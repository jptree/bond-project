import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
# import altair as alt
import datetime

# D:\Python\Python38\Scripts\streamlit run main.py
# D:\Python\Python38\Scripts\streamlit run D:\Python\Projects\bondProjectLocal\main.py
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


def industry_filter():
    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    data = df

    container = st.container()
    all = st.checkbox("Select all")

    if all:
        industry_options = container.multiselect("Choose an industry",
                                                 list(df['industrySector'].value_counts().index),
                                                 list(df['industrySector'].value_counts().index))
    else:
        industry_options = container.multiselect("Choose an industry",
                                                 list(df['industrySector'].value_counts().index))

    if not industry_options:
        st.error("Select at least one industry")
    else:
        data = data[data['industrySector'].isin(industry_options)]
        chart = st.line_chart(cumulative_pnl(data))


def chart_time_window():
    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # age = st.slider('How old are you?', 0, 130, 25)
    # st.write("I'm ", age, 'years old')


    # Add time frequency?
    time_window = st.slider(
        "Select chart time window range:",
        value=(df['timestamp'].min().to_pydatetime(), df['timestamp'].max().to_pydatetime()),
        min_value=df['timestamp'].min().to_pydatetime(),
        max_value=df['timestamp'].max().to_pydatetime(),
        step=datetime.timedelta(0, 60 * 5),
        format='D-M-YYYY H:m'
    )

    start_time = time_window[0]
    end_time = time_window[1]
    data = df[(df['timestamp'] > start_time) & (df['timestamp'] < end_time)]
    chart = st.line_chart(cumulative_pnl(data))

    n_cusip = 5


    n_cusip_input = st.text_input(
        label='Enter number of CUSIPS desired in table below',
        value='5'
    )

    try:
        n_cusip = int(n_cusip_input)
    except ValueError:
        st.error('Please enter a valid integer')
        n_cusip = 5

    winners, losers = cumulative_change_window_gains_losses(n_cusip, data)

    winners = winners[['difference_since_beginning', 'cusip']]
    winners.index = np.arange(1, len(winners) + 1)
    winners.columns = ['Profits', 'CUSIP']

    losers = losers[['difference_since_beginning', 'cusip']]
    losers.index = np.arange(1, len(losers) + 1)
    losers.columns = ['Losses', 'CUSIP']


    st.header('Total P&L from Beginning to End of Period')
    col1, spacing, col2 = st.columns([5, 1, 5])
    with col1:
        # st.header('Total P&L from BOP to EOP: Gain')
        st.table(winners)
    with col2:
        # st.header('Total P&L from BOP to EOP: Loss')
        st.table(losers)


def main():
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Winners", "Home phone", "Mobile phone")
    )




    chart_time_window()


if __name__ == "__main__":

    # df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # data = df

    # print(date_window('2021-07-28 20:55:00', '2021-07-28 21:55:00', df))
    # print(winners_losers(5, df))
    # print(deviation_bands(1, 5, df))
    # print(cumulative_change_window_gains_losses(30, df)[1])
    # chart = st.line_chart(cumulative_pnl(data))

    # chart_time_window()
    main()
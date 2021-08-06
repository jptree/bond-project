def intro():
    import streamlit as st

    st.sidebar.success("Select a view above.")

    st.markdown(
        """
        This application was developed by Jason F. Petri. There are various views to select from that all help in the
        analysis of the securities provided.
        
        *August 6, 2021*
    """
    )


def pnl_cusip_attribution():
    import streamlit as st
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from main import cumulative_pnl, cumulative_change_window_gains_losses

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
        step=timedelta(0, 60 * 5),
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
        st.text('List of all CUSIPS')
        st.text(' '.join(list(winners['CUSIP'])))
    with col2:
        # st.header('Total P&L from BOP to EOP: Loss')
        st.table(losers)
        st.text('List of all CUSIPS')
        st.text(' '.join(list(losers['CUSIP'])))


def individual_cusip_analysis():
    import streamlit as st
    import pandas as pd
    from datetime import timedelta
    from main import cumulative_pnl
    import numpy as np

    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    desired_cusips_input = st.text_input(
        label='Enter desired CUSIPS separated by a space',
        value='928563AJ4 285512AE9 254709AM0 760759AU4 92826CAP7'
    )

    desired_cusips = desired_cusips_input.split(' ')

    time_window = st.slider(
        "Select chart time window range:",
        value=(df['timestamp'].min().to_pydatetime(), df['timestamp'].max().to_pydatetime()),
        min_value=df['timestamp'].min().to_pydatetime(),
        max_value=df['timestamp'].max().to_pydatetime(),
        step=timedelta(0, 60 * 5),
        format='D-M-YYYY H:m'
    )

    start_time = time_window[0]
    end_time = time_window[1]
    data = df[(df['timestamp'] > start_time) & (df['timestamp'] < end_time)]
    data = data[data['cusip'].isin(desired_cusips)]

    other_cusip_info = data.drop_duplicates('cusip')[['cusip', 'type', 'benchmark_cusip', 'ticker', 'securitydes',
                                                      'industrySector', 'mat_bucket', 'liq_score', 'liq_bucket', 'weightedage']].set_index('cusip')

    # chart = st.line_chart({'a': [1, 2, 3], 'b': [2, 2, 4]})

    def multi_plot(data):
        data = data.groupby(['cusip', 'timestamp']).mean()[['spread_pnl']]
        data = data.unstack().T.reset_index().set_index('timestamp').drop(columns=['level_0'])
        data = data.fillna(0)
        return np.cumsum(data)

    # print(multi_plot(data))
    chart = st.line_chart(multi_plot(data).to_dict())

    st.header('Characteristics of desired cusips')
    st.table(other_cusip_info)

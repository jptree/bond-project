def intro():
    import streamlit as st

    st.sidebar.success("Select a view above.")

    st.markdown(
        """
        This application was developed by Jason F. Petri. There are various views to select from--all of which help
        in the analysis of the securities provided.
        
        *August 10, 2021*
        """
    )


def pnl_cusip_attribution():
    import streamlit as st
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from main import cumulative_pnl, cumulative_change_window_gains_losses, cumulative_winners_losers_new

    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # df['cumulative_pnl'] = cumulative_pnl(df).reset_index()
    cumulative = df.groupby(['cusip', 'timestamp'])['spread_pnl'].sum().groupby(level=0).cumsum().reset_index()
    cumulative.columns = ['cusip', 'timestamp', 'cumulative_pnl']
    df = pd.merge(df, cumulative, on=['timestamp', 'cusip'])


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
    data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    chart = st.line_chart(cumulative_pnl(data))

    n_cusip_input = st.text_input(
        label='Enter number of CUSIPS desired in table below',
        value='5'
    )

    try:
        n_cusip = int(n_cusip_input)
    except ValueError:
        st.error('Please enter a valid integer')
        n_cusip = 5

    winners, losers = cumulative_winners_losers_new(data, n_cusip)

    winners = winners.reset_index()
    winners.index = np.arange(1, len(winners) + 1)
    winners.columns = ['CUSIP', 'Profits']

    losers = losers.reset_index()
    losers.index = np.arange(1, len(losers) + 1)
    losers.columns = ['CUSIP', 'Losses']


    st.header('Total P&L from Beginning to End of Period')
    col1, spacing, col2 = st.columns([5, 1, 5])
    with col1:
        st.table(winners)
        st.text('List of all CUSIPS')
        st.text(' '.join(list(winners['CUSIP'])))
    with col2:
        st.table(losers)
        st.text('List of all CUSIPS')
        st.text(' '.join(list(losers['CUSIP'])))


def individual_cusip_analysis():
    import streamlit as st
    import pandas as pd
    from datetime import timedelta
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cumulative = df.groupby(['cusip', 'timestamp'])['spread_pnl'].sum().groupby(level=0).cumsum().reset_index()
    cumulative.columns = ['cusip', 'timestamp', 'cumulative_pnl']
    df = pd.merge(df, cumulative, on=['timestamp', 'cusip'])


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
    data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    data = data[data['cusip'].isin(desired_cusips)]

    other_cusip_info = data.drop_duplicates('cusip')[['cusip', 'type', 'benchmark_cusip', 'ticker', 'securitydes',
                                                      'industrySector', 'mat_bucket', 'liq_score', 'liq_bucket', 'weightedage']].set_index('cusip')

    cusip_data = data.groupby(['cusip', 'timestamp'])['cumulative_pnl'].sum()

    normalize = st.checkbox(label='Normalize data')

    plot_data = {}
    for c in desired_cusips:
        if normalize:
            plot_data[c] = cusip_data[c] - cusip_data[c][0]
        else:
            plot_data[c] = cusip_data[c]

    chart = st.line_chart(plot_data)


    st.header('Characteristics of desired cusips')
    st.table(other_cusip_info)

    st.header('P&L Change Distribution')
    selected_cusip = st.selectbox(label='Select one of your CUSIPs', options=desired_cusips)
    number_of_bins = st.slider(label='Select the number of bins', min_value=10, max_value=100, value=40)
    selected_cusip_data = data[data['cusip'] == selected_cusip]['spread_pnl']

    fig, ax = plt.subplots()
    ax.hist(selected_cusip_data, number_of_bins, density=False)
    ax.set_xlabel('P&L')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Frequency of P&L for {selected_cusip}, {number_of_bins} Bins')

    st.pyplot(fig)


def bucket_industry():
    import streamlit as st
    import pandas as pd
    from datetime import timedelta
    from numpy import cumsum
    from main import cumulative_pnl, cumulative_change_window_gains_losses, cumulative_winners_losers_new

    # def select_all(column, key, all, label):
    #     if all:
    #         return st.multiselect(
    #             label=label,
    #             options=df[column].unique(),
    #             default=df[column].unique(),
    #             key=key
    #         )
    #     else:
    #         return st.multiselect(
    #             label=label,
    #             options=df[column].unique()
    #         )


    def filter(column, selected, data):
        return data[data[column].isin(selected)]


    df = pd.read_csv('ctap_analytics_pnl_decomp_sample_day.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])



    cumulative = df.groupby(['cusip', 'timestamp'])['spread_pnl'].sum().groupby(level=0).cumsum().reset_index()
    cumulative.columns = ['cusip', 'timestamp', 'cumulative_pnl']
    df = pd.merge(df, cumulative, on=['timestamp', 'cusip'])
    data = df.copy()

    st.header('Filter universe of securities below')

    industry_filter = st.multiselect(label='Select industries', options=df['industrySector'].unique(), default=df['industrySector'].unique())
    # industry_all = st.checkbox("Select all", key='industry')
    # industry_filter = select_all('industrySector', 'industryFilter', industry_all, 'Select industries')
    if not industry_filter:
        st.error('Select at least one industry!')
    else:
        data = filter('industrySector', industry_filter, data)

    benchmark_filter = st.multiselect(label='Select benchmark CUSIPs', options=df['benchmark_cusip'].unique(),
                                     default=df['benchmark_cusip'].unique())
    # benchmark_all = st.checkbox("Select all", key='benchmark')
    # benchmark_filter = select_all('benchmark_cusip', 'benchmarkFilter', benchmark_all, 'Select benchmark CUSIPs')
    if not benchmark_filter:
        st.error('Select at least one benchmark CUSIPs!')
    else:
        data = filter('benchmark_cusip', benchmark_filter, data)

    maturity_filter = st.multiselect(label='Select maturity buckets', options=df['mat_bucket'].unique(),
                                     default=df['mat_bucket'].unique())
    # maturity_all = st.checkbox("Select all", key='maturity')
    # maturity_filter = select_all('mat_bucket', 'maturityFilter', maturity_all, 'Select maturity buckets')
    if not maturity_filter:
        st.error('Select at least one maturity bucket!')
    else:
        data = filter('mat_bucket', maturity_filter, data)

    liquidity_filter = st.multiselect(label='Select liquidity buckets', options=df['liq_bucket'].unique(),
                                     default=df['liq_bucket'].unique())
    # liquidity_all = st.checkbox("Select all", key='liquidity')
    # liquidity_filter = select_all('liq_bucket', 'liquidityFilter', liquidity_all, 'Select liquidity buckets')
    if not liquidity_filter:
        st.error('Select at least one liquidity bucket!')
    else:
        data = filter('liq_bucket', liquidity_filter, data)

    # st.table(data)

    st.header('Plot of P&L performance')
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
    # data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
    sorting_options = ['industrySector', 'benchmark_cusip', 'mat_bucket', 'liq_bucket']
    selected_sort = st.selectbox(label='Select the characteristic you want to plot on', options=sorting_options)
    normalize = st.checkbox(label='Normalize data')

    plot_data = cumsum(data.groupby([selected_sort, 'timestamp'])['spread_pnl'].sum().unstack().T.fillna(0))
    plot_data = plot_data[(plot_data.index >= start_time) & (plot_data.index <= end_time)]
    if len(plot_data) != 0:

        plot = {}
        for s in data[selected_sort].unique():
            if normalize:

                plot[s] = plot_data[s] - plot_data[s][0]
            else:
                plot[s] = plot_data[s]

        chart = st.line_chart(plot)
    else:
        st.error('No data available with the current combination of filters!')

    # change_pnl = data.groupby([selected_sort, 'timestamp'])['cumulative_pnl'].sum().reset_index()
    # change_pnl = change_pnl.groupby([selected_sort])['cumulative_pnl'].last() - change_pnl.groupby([selected_sort])['cumulative_pnl'].first()
    # change_pnl = change_pnl.reset_index()
    # change_pnl.columns = [selected_sort, 'Cumulative P&L Difference']
    # change_pnl = change_pnl.sort_values(by='Cumulative P&L Difference').reset_index()[[selected_sort, 'Cumulative P&L Difference']]
    # # st.table([change_pnl[change_pnl])
    # st.table(change_pnl)

    change_pnl = plot_data.iloc[-1] - plot_data.iloc[0]
    change_pnl = change_pnl.sort_values().reset_index()
    change_pnl.columns = [selected_sort, 'Cumulative P&L Difference']
    st.table(change_pnl)


def merchant_extraction():
    import streamlit as st
    from main import load_and_use_model
    from PIL import Image
    import numpy as np

    sample_inputs = ['DISCOUNT DRUG MART 83 MEDINA OH',
                     'AKRONYM BREWING LLC AKRON OH',
                     'STEAK-N-SHAKE#0578 Q99 BRUNSWICK OH',
                     'BUFFALO WILD WINGS MED STRONGSVILLE OH',
                     'CHICK-FIL-A #01920 FAIRLAWN OH',
                     'APPLE STORE #45 FRESNO CA']

    # sample_choice = random.choice(sample_inputs)
    sample_outputs = ['DISCOUNT DRUG MART', 'AKRONYM BREWING', 'STEAK-N-SHAKE', 'BUFFALO WILD WINGS', 'CHICK-FIL-A',
                      'APPLE STORE #']


    raw_input = st.text_input(label='Enter a raw transaction statement', value='CHICK-FIL-A #01920 FAIRLAWN OH')
    st.text(f'Output: {load_and_use_model("models/20210809-120113", [raw_input])}')
    st.header('Sample transaction inputs and outputs')
    st.table({'Inputs': sample_inputs, 'Outputs': sample_outputs})

    st.header('What is a convolutional neural network?')
    st.image(np.array(Image.open('cnn_image.jpeg')))

    st.header('Full disclosures...')
    st.markdown(
        """
        This *current* model will not be fantastic (on data it has not seen yet) for one big reason:
        I need more data! I am not a big time shopper, and these neural network models need data to function. The model
        will only be able to perform well if it has learned patterns from previous training data. This model was trained
        on my banking history and my parents (they do more shopping than I do!). All in, I had about 300 records that
        I manually classified (dictating where the merchant name is). The model could near human levels of performance
        with a larger training set and more trainable parameters.
        """

    )
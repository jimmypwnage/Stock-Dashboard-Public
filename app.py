import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import sharpe_ratio_pkg as sharpe
from scipy import stats
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.graph_objects as go
import random
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.environ.get("access-key")

cols = DEFAULT_PLOTLY_COLORS

end_date_default = datetime.datetime.today().strftime('%Y-%m-%d')
start_date_default = (datetime.datetime.today() - datetime.timedelta(weeks=12)).strftime('%Y-%m-%d')

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.title = 'Portfolio Optimizer'

debug_mode = False
num_options = 5

header_card = html.Div(
    children=[
        html.P(children='📈💻', className='header-emoji'),
        html.H1(children='Portfolio Optimizer', className='header-title'),
        html.P(
            children='Optimize a given portfolio weightage using Sharpe Ratio',
            className='header-description'
            )
        ],
    className='header'
)

current_portfolio = {
    'TSLA': 1.4,
    'AMC': 1,
    'GME': 2,
    'MSFT': 1,
    'AAPL': 5
}


def query_stock(ticker, date_from, date_to):
    stock_data = yf.download(
        tickers=ticker,
        group_by='ticker',
        start=date_from,
        end=date_to,
        threads=True
    )

    if stock_data.shape[0] == 0:
        return pd.DataFrame()
    elif ticker[0] == '^':
        stock_data['symbol'] = ticker + '.INDX'
        stock_data['exchange'] = 'INDX'
    else:
        stock_data['symbol'] = ticker
        stock_data['exchange'] = 'EQUITY'

    stock_data.reset_index(inplace=True)
    stock_data.sort_values('Date', inplace=True)

    stock_data.rename(columns={'Date': 'date',
                               'Adj Close': 'adj_close',
                               'Volume': 'final_volume'}, inplace=True)

    return stock_data


def generate_options_card(num_options):
    options_card = []

    for i in range(num_options):
        # handle titles
        if i == 0:
            stock_card = [
                html.Div(children='Stock Ticker', className='menu-title'),
                dcc.Input(
                    id=f'stock_name_input_{i}',
                    type='search',
                    placeholder='Input Ticker here',
                    value=list(current_portfolio)[i],
                    className='input',
                    multiple=True
                )
            ]
            value_input_card = [
                html.Div(children='Units Bought', className='menu-title'),
                dcc.Input(
                    id=f'initial_amt_{i}',
                    type='number',
                    placeholder=1,
                    value=list(current_portfolio.values())[i],
                    className='input'
                )
            ]
            date_range_card = [
                html.Div(children='Date Range', className='menu-title'),
                dcc.DatePickerRange(
                    id=f'date_range_{i}',
                    min_date_allowed=datetime.date(2010, 1, 1),
                    max_date_allowed=datetime.datetime.strptime(end_date_default, '%Y-%m-%d'),
                    start_date=datetime.datetime.strptime(start_date_default, '%Y-%m-%d'),
                    end_date=datetime.datetime.strptime(end_date_default, '%Y-%m-%d'),
                    display_format='YYYY MMM DD',
                    className='input-date'
                )
            ]
        else:
            stock_card = [
                dcc.Input(
                    id=f'stock_name_input_{i}',
                    type='search',
                    placeholder='Input Ticker here',
                    value=list(current_portfolio)[i],
                    className='input',
                    multiple=True
                )
            ]
            value_input_card = [
                dcc.Input(
                    id=f'initial_amt_{i}',
                    type='number',
                    placeholder=1,
                    value=list(current_portfolio.values())[i],
                    className='input'
                )
            ]
            date_range_card = [
                dcc.DatePickerRange(
                    id=f'date_range_{i}',
                    min_date_allowed=datetime.date(2010, 1, 1),
                    max_date_allowed=datetime.datetime.strptime(end_date_default, '%Y-%m-%d'),
                    start_date=datetime.datetime.strptime(start_date_default, '%Y-%m-%d'),
                    end_date=datetime.datetime.strptime(end_date_default, '%Y-%m-%d'),
                    display_format='YYYY MMM DD',
                    className='input-date'
                )
            ]

        if i == num_options - 1:
            search_button = [dbc.Button('Search', id=f'search_button', block=True, n_clicks=1,
                                        color='success', className='input-search')]
        else:
            search_button = []

        options_card_temp = html.Div(
            children=[
                html.Div()
                ,
                html.Div(
                    children=stock_card
                ),
                html.Div(
                    children=value_input_card
                ),
                html.Div(
                    children=date_range_card
                ),
                html.Div(
                    children=search_button
                ),

            ],
            className='menu',
        )

        options_card.append(options_card_temp)

    return options_card


graphs_card = html.Div(
          children=[
              html.Div(
                  children=dcc.Graph(
                      id='amount-chart', config={'displayModeBar': False},
                      style={'width': '80wh', 'height': '100vh'}
                  ),
                  className='card',
              ),
              html.Div(children=[
                  html.Div(children='Alpha and Beta of Portfolio to Indexes', className='menu-card-title-alpha-beta'),
                  html.Div(id='alpha-beta-container',
                           children=[
                                  html.Div(id='greeks-index-1', className='card-for-alpha-beta'),
                                  html.Div(id='greeks-index-2', className='card-for-alpha-beta'),
                                  html.Div(id='greeks-index-3', className='card-for-alpha-beta')
                            ], style={'columnCount': 3})
              ], className='container-alpha-beta'
              ),
              html.Div(
                  children=[
                      html.Div(children='Select type of graph', className='menu-card-title'),
                      dcc.Dropdown(
                          id='all-stock-chart-choice',
                          options=[
                              {'label': x, 'value': x}
                              for x in ['Stocks Raw Price vs Index',
                                        'Stocks Normalized vs Index',
                                        'Portfolio Normalized vs Index']],
                          value='Portfolio Normalized vs Index',
                          clearable=False,
                          className='dropdown'
                      ),
                      dcc.Graph(
                          id='all-stock-chart', config={'displayModeBar': False},
                          style={'width': '80wh', 'height': '65vh'}
                      )
                  ],
                  className='card-with-dropdown',
              ),
              html.Div(
                  children=dcc.Graph(
                      id='subplot-chart', config={'displayModeBar': False},
                      style={'width': '80wh', 'height': '80vh'}
                  ),
                  className='card',
              ),
              html.Div(
                  children=dcc.Graph(
                      id='efficient-frontier-chart', config={'displayModeBar': False},
                      style={'width': '80wh', 'height': '80vh'}
                  ),
                  className='card',
              ),
          ],
          className='wrapper'
        )

options_card = generate_options_card(num_options)

layout_children = [
        # header
        header_card,
        html.Div(id='alert_div', children=[]),
        html.Div(id='query_df', style={'display': 'none'}),
        graphs_card,
    ]

layout_children[1:1] = options_card

app.layout = html.Div(
    children=layout_children,
)

input_list = []

for i in range(num_options):
    input_list.append(State(f'stock_name_input_{i}', 'value'))
    input_list.append(State(f'initial_amt_{i}', 'value'))
    input_list.append(State(f'date_range_{i}', 'start_date'))
    input_list.append(State(f'date_range_{i}', 'end_date'))


@app.callback(
    [
        Output('alert_div', 'children'),
        Output('query_df', 'children')
    ],
    [
        Input('search_button', 'n_clicks')
    ],
    input_list
)
def query_data(*args):
    n_clicks = args[0]

    ticker_index = [index_num for index_num in range(len(args)) if (index_num - 1) % 4 == 0]
    stock_tickers = [args[index_num] for index_num in ticker_index]
    num_stocks = [args[index_num + 1] for index_num in ticker_index]
    start_date = [args[index_num + 2] for index_num in ticker_index]
    end_date = [args[index_num + 3] for index_num in ticker_index]

    interested_df = pd.DataFrame(
        {
            'stock_ticker': stock_tickers,
            'num_stocks': num_stocks,
            'start_date': start_date,
            'end_date': end_date
        }
    )

    interested_df['start_date'] = pd.to_datetime(interested_df['start_date'])
    interested_df['end_date'] = pd.to_datetime(interested_df['end_date'])

    earliest_start_date = interested_df['start_date'].min().strftime('%Y-%m-%d')
    latest_end_date = interested_df['end_date'].max().strftime('%Y-%m-%d')

    interested_df['start_date'] = interested_df['start_date'].dt.strftime('%Y-%m-%d')
    interested_df['end_date'] = interested_df['end_date'].dt.strftime('%Y-%m-%d')

    if debug_mode:
        interested_df = pd.read_csv('interested_df.csv')

    print(f'Searched {n_clicks} time(s) at {datetime.datetime.now()}')

    # query data
    ''' To do: think of logic to prevent redundant re-querying:
        If only value change, no need to re-query
        If ticker change, only re-query the changed ticker,
        If date range change, check whether new date range is within the old date range
    '''
    if debug_mode:
        stock_data = pd.read_csv('stock_data.csv')
    else:
        result_list = [query_stock(row[0], row[1], row[2]) for row in zip(interested_df['stock_ticker'],
                                                                          interested_df['start_date'],
                                                                          interested_df['end_date'])]
        stock_data = pd.concat(result_list)

    # query indexes (S&P 500 & DJI)
    indexes_list = ['^GSPC', '^IXIC', '^DJI']

    for market_index in indexes_list:
        stock_data = stock_data.append(query_stock(market_index, earliest_start_date, latest_end_date))

    # check which ticker has error
    query_list = set(interested_df['stock_ticker'].unique())
    result_list = set(stock_data['symbol'].unique())

    difference_string = ', '.join(set(query_list).difference(result_list))

    if len(difference_string) > 0:
        alert = dbc.Alert(f'The following ticker(s): {difference_string} does not exist. '
                          'Please search on Yahoo Finance for the tickers supported',
                          color='danger', dismissable=False, className='alert-bar')
    else:
        alert = None

    first_row = stock_data.groupby(['symbol']).first()[['date', 'adj_close']]
    initial_data = interested_df[['stock_ticker', 'num_stocks',
                                  'start_date', 'end_date']].merge(first_row,
                                                                   left_on='stock_ticker',
                                                                   right_on=first_row.index)
    initial_data['unit_bought'] = initial_data['num_stocks']
    initial_data['value_invested'] = initial_data['unit_bought'] * initial_data['adj_close']
    initial_data = initial_data[['stock_ticker', 'date', 'end_date', 'value_invested', 'unit_bought']]
    initial_data.rename(columns={'date': 'start_date'}, inplace=True)

    combined_data = stock_data[['symbol', 'exchange', 'adj_close', 'date']].merge(initial_data,
                                                                                          left_on='symbol',
                                                                                          right_on='stock_ticker',
                                                                                          how='left')
    combined_data['total_value'] = combined_data['adj_close'] * combined_data['unit_bought']

    return alert, [combined_data.to_json(date_format='iso', orient='split'),
                   initial_data.to_json(date_format='iso', orient='split')]


@app.callback(
    [
        Output('amount-chart', 'figure'),
        Output('subplot-chart', 'figure')
    ],
    [
        Input('query_df', 'children')
    ]
)
def update_charts(jsonified_cleaned_data):
    combined_data = pd.read_json(jsonified_cleaned_data[0], orient='split')

    grouped_data = combined_data.groupby(['date'])['total_value'].sum().to_frame()

    daily_returns_overall = grouped_data['total_value'].pct_change() * 100

    amount_chart_figure = make_subplots(rows=2, cols=1,
                                        subplot_titles=[f'Total Amount ($)',
                                                        f'Daily % Change'
                                                        ],
                                        shared_xaxes='all',
                                        horizontal_spacing=0.05,
                                        vertical_spacing=0.10)

    amount_chart_figure.append_trace(
        go.Scatter(x=daily_returns_overall.index, y=daily_returns_overall,
                   line={'color': '#FFCCCB'},
                   showlegend=False,
                   hovertemplate='%{y:.2f}%<extra></extra>'),
        row=2,
        col=1
    )

    non_index_stocks = combined_data[combined_data['exchange'] != 'INDX']

    performance_fig = make_subplots(rows=2, cols=non_index_stocks['symbol'].nunique(),
                                    subplot_titles=[stock_name for stock_name in non_index_stocks['symbol'].unique()],
                                    shared_xaxes='all',
                                    horizontal_spacing=0.05,
                                    vertical_spacing=0.10)

    num_stocks = non_index_stocks['symbol'].nunique()

    for j in range(num_stocks):
        stock_name = non_index_stocks['symbol'].unique()[j]
        temp_stock = non_index_stocks[non_index_stocks['symbol'] == stock_name]

        daily_returns = temp_stock.set_index('date')['adj_close'].pct_change() * 100

        amount_chart_figure.append_trace(
            go.Scatter(x=temp_stock['date'], y=temp_stock['total_value'],
                       mode='lines',
                       stackgroup='one',
                       name=stock_name,
                       text=temp_stock['symbol'],
                       line=dict(color=cols[j]),
                       hovertemplate='%{text}: $%{y:.2f}<extra></extra>'),
            row=1,
            col=1
        )

        performance_fig.append_trace(
            go.Scatter(x=temp_stock['date'], y=temp_stock['adj_close'],
                       hovertemplate='$%{y:.2f}<extra></extra>',
                       line=dict(color=cols[j]),
                       opacity=0.5
                       ),
            row=1,
            col=j + 1
        )

        performance_fig.append_trace(
            go.Scatter(x=daily_returns.index, y=daily_returns,
                       hovertemplate='%{y:.2f}%<extra></extra>',
                       line=dict(color=cols[j]),
                       opacity=0.5
                       ),
            row=2,
            col=j + 1,
        )

    amount_chart_figure.append_trace(
        go.Scatter(x=grouped_data.index, y=grouped_data['total_value'],
                   showlegend=False,
                   line=dict(color='rgba(46, 49, 49, 1)'),
                   hovertemplate='<b>Total: $%{y:.2f}</b><extra></extra>'),
        row=1,
        col=1
    )

    amount_chart_figure.update_layout(template='plotly_white', showlegend=True, hovermode='x unified',
                                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                      title_text=f'Overall Performance for '
                                                 f'All Stocks ({", ".join(non_index_stocks["symbol"].unique())})')

    amount_chart_figure.update_yaxes(title_text='Price ($)', row=1, tickprefix='$')
    amount_chart_figure.update_yaxes(title_text='Daily % Change', row=2, ticksuffix='%')

    performance_fig.update_layout(template='plotly_white', showlegend=False, hovermode='x unified',
                                  title_text='Performance of individual stocks')
    performance_fig.update_yaxes(title_text='Price ($)', row=1, col=1)
    performance_fig.update_yaxes(row=1, tickprefix='$')
    performance_fig.update_yaxes(title_text='Daily % Change', row=2, col=1)
    performance_fig.update_yaxes(row=2, ticksuffix='%')

    return amount_chart_figure, performance_fig


@app.callback(
    [
        Output('all-stock-chart', 'figure'),
        Output('greeks-index-1', 'children'),
        Output('greeks-index-2', 'children'),
        Output('greeks-index-3', 'children')
    ],
    [
        Input('query_df', 'children'),
        Input('all-stock-chart-choice', 'value')
    ]
)
def update_performance_all(query_df, type_of_graph):
    combined_data = pd.read_json(query_df[0], orient='split')

    non_index_stocks = combined_data[combined_data['exchange'] != 'INDX']
    grouped_portfolio = non_index_stocks.groupby(['date'])['total_value'].sum().to_frame().reset_index()
    grouped_portfolio['symbol'] = 'PORTFOLIO'
    grouped_portfolio = grouped_portfolio[['date', 'symbol', 'total_value']]

    index_stocks = combined_data[combined_data['exchange'] == 'INDX'].copy()
    index_stocks.loc[:, 'total_value'] = index_stocks.loc[:, 'adj_close']
    index_stocks = index_stocks[['date', 'symbol', 'total_value']]

    overall_plot = go.Figure()

    if type_of_graph in ['Stocks Raw Price vs Index', 'Stocks Normalized vs Index']:
        num_stocks = combined_data['symbol'].nunique()

        for stock_num in range(num_stocks):
            stock_name = combined_data['symbol'].unique()[stock_num]
            temp_stock = combined_data[combined_data['symbol'] == stock_name]

            if temp_stock['exchange'].unique()[0] == 'INDX':
                dash_value = 'dashdot'
            else:
                dash_value = 'solid'

            initial_price = temp_stock['adj_close'].iloc[0]

            if type_of_graph == 'Stocks Raw Price vs Index':
                overall_plot.add_trace(
                    go.Scatter(x=temp_stock['date'], y=temp_stock['adj_close'],
                               name=stock_name, hovertemplate='$%{y:.2f}',
                               line=dict(color=cols[stock_num], dash=dash_value)
                               )
                )

                overall_plot.update_yaxes(title_text='Price ($)', tickprefix='$')
            elif type_of_graph == 'Stocks Normalized vs Index':
                overall_plot.add_trace(
                    go.Scatter(x=temp_stock['date'], y=1000 * temp_stock['adj_close'] / initial_price,
                               name=stock_name, hovertemplate='%{y:.0f}',
                               line=dict(color=cols[stock_num],  dash=dash_value),
                               )
                )

                overall_plot.update_yaxes(title_text='Price Normalized')
    elif type_of_graph == 'Portfolio Normalized vs Index':
        master_data = pd.concat([grouped_portfolio, index_stocks])

        num_stocks = master_data['symbol'].nunique()

        for stock_num in range(num_stocks):
            stock_name = master_data['symbol'].unique()[stock_num]
            temp_stock = master_data[master_data['symbol'] == stock_name]
            initial_price = temp_stock['total_value'].iloc[0]

            if stock_name not in index_stocks['symbol'].unique():
                overall_plot.add_trace(
                    go.Scatter(x=temp_stock['date'], y=1000 * temp_stock['total_value'] / initial_price,
                               name=stock_name, hovertemplate='%{y:.0f}',
                               line=dict(color='rgba(46, 49, 49, 1)', width=2)
                               )
                )
            else:
                overall_plot.add_trace(
                    go.Scatter(x=temp_stock['date'], y=1000 * temp_stock['total_value'] / initial_price,
                               name=stock_name, hovertemplate='%{y:.0f}',
                               line=dict(color=cols[stock_num],  dash='dashdot')
                               )
                )

        overall_plot.update_yaxes(title_text='Price Normalized')

    overall_plot.update_layout(template='plotly_white', hovermode='x unified',
                               legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                           xanchor='right', x=1),
                               title_text='All stock(s) historical prices compared to indexes')

    # run alpha-beta analysis
    daily_returns_overall = grouped_portfolio['total_value'].pct_change()[1:]

    index_table = pd.pivot_table(data=index_stocks, index='date', columns='symbol', values='total_value', aggfunc='sum')
    index_returns = index_table.pct_change()[1:]

    output_perf_list = [overall_plot]

    for index_stock_name in index_stocks['symbol'].unique():
        beta, alpha, r_value = stats.linregress(index_returns[index_stock_name].values,
                                                daily_returns_overall.values)[:3]
        output_perf_list.append(dcc.Markdown(f"""
                                    **{index_stock_name}**

                                    Alpha α: {round(alpha * 100, 3)}%  
                                    Beta β: {round(beta, 5)}  
                                    R-squared: {round(r_value**2, 2)}
                                """))

    return output_perf_list


@app.callback(
    [
        Output('efficient-frontier-chart', 'figure'),
     ],
    [
        Input('query_df', 'children')
    ]
)
def update_efficient_frontier(jsonified_cleaned_data):
    combined_data = pd.read_json(jsonified_cleaned_data[0], orient='split')

    non_index_stocks = combined_data[combined_data['exchange'] != 'INDX']

    # get efficient frontier for sharpe ratio
    table = pd.pivot_table(data=non_index_stocks, index='date', columns='symbol', values='adj_close', aggfunc='sum')
    returns = table.pct_change()

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    risk_free_rate = 0.01252

    # get maximum return
    max_ret = sharpe.max_return(mean_returns, cov_matrix)
    sdp_max, rp_max = sharpe.portfolio_annualized_performance(max_ret['x'], mean_returns, cov_matrix)

    # get minimum volatility
    min_vol = sharpe.min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = sharpe.portfolio_annualized_performance(min_vol['x'], mean_returns, cov_matrix)

    # get maximum sharpe ratio
    max_sharpe = sharpe.max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = sharpe.portfolio_annualized_performance(max_sharpe['x'], mean_returns, cov_matrix)

    # get the efficiency frontier
    target = np.linspace(rp_min, rp_max, 100)
    efficient_portfolios = sharpe.efficient_frontier(mean_returns, cov_matrix, target)
    weights = [p['x']*100 for p in efficient_portfolios]
    volatility = [p['fun'] for p in efficient_portfolios]
    efficient_df = pd.DataFrame({'weight': weights,
                                 'volatility': volatility,
                                 'returns': target
                                 })

    initial_data = pd.read_json(jsonified_cleaned_data[1], orient='split')

    # get current portfolio performance
    initial_data['weight'] = initial_data['value_invested'] / initial_data['value_invested'].sum()
    cur_weight = initial_data.groupby('stock_ticker')['weight'].sum()
    cur_port_std, cur_port_ret = sharpe.portfolio_annualized_performance(cur_weight,
                                                                         mean_returns,
                                                                         cov_matrix)

    print('Done')

    list_of_stocks = mean_returns.index.tolist()
    efficient_df[list_of_stocks] = pd.DataFrame(efficient_df['weight'].tolist(),
                                                index=efficient_df.index)
    an_rt = mean_returns * 252
    an_vol = np.std(returns) * np.sqrt(252)

    sharpe_ratio_ind = (an_rt - risk_free_rate) / an_vol

    hover_string_frontier = ''
    hover_string_cur = ''
    for k in range(len(list_of_stocks)):
        hover_string_frontier += f'{mean_returns.index[k]}: %{{customdata[{k}]:.2f}}%<br>'
        hover_string_cur += f'{cur_weight.index[k]}: {round(cur_weight[k] * 100, 2)}% <br>'

    sharpe_ratio_fig = go.Figure()

    sharpe_ratio_fig.add_trace(
        go.Scatter(
            x=efficient_df['volatility'], y=efficient_df['returns'],
            text=((efficient_df['returns'] - risk_free_rate) / efficient_df['volatility']).round(3),
            customdata=efficient_df[list_of_stocks],
            hoverlabel={'namelength': -1},
            hovertemplate='<b>Volatility: %{x:.2f} <br>' +
                          'Returns: %{y:.2f}<br>' +
                          'Sharpe Ratio: %{text}</b><br><br>' +
                          '<b>Weight</b>:<br>' +
                          hover_string_frontier + '<extra></extra>',
            line=dict(color='lightblue')
        )
    )

    sharpe_ratio_fig.add_trace(
        go.Scatter(
            x=[sdp], y=[rp],
            mode='markers',
            marker_symbol='star',
            marker=dict(color='red'),
            marker_size=20,
            hoverinfo='skip'
        )
    )

    sharpe_ratio_fig.add_trace(
        go.Scatter(
            x=an_vol, y=an_rt,
            mode='markers',
            marker=dict(color=['lightgreen' if val >= 0 else '#FFCCCB' for val in sharpe_ratio_ind]),
            marker_size=[abs(val) * 15 if abs(val) * 15 >= 5 else 20 for val in sharpe_ratio_ind],
            text=an_rt.index,
            hovertemplate='<b>%{text}</b><br><br>' +
                          'Volatility: %{x:.2f} <br>' +
                          'Returns: %{y:.2f}<extra></extra>'

        )
    )

    # current portfolio weightage
    sharpe_ratio_fig.add_trace(
        go.Scatter(
            x=[cur_port_std], y=[cur_port_ret],
            mode='markers',
            marker_symbol='x',
            marker=dict(color='#d3d3d3'),
            text=[hover_string_cur],
            marker_size=30,
            hovertemplate='<b>Current Portfolio</b><br>' +
                          'Volatility: %{x:.2f} <br>' +
                          'Returns: %{y:.2f}<br><br>' +
                          '<b>Weight</b><br>%{text}<extra></extra>'

        )
    )

    sharpe_ratio_fig.update_layout(template='plotly_white', showlegend=False,
                                   title_text='Optimum Frontier for Sharpe Ratio')
    sharpe_ratio_fig.update_yaxes(title_text='Annualized Returns')
    sharpe_ratio_fig.update_xaxes(title_text='Annualized Volatility')

    return [sharpe_ratio_fig]


if __name__ == '__main__':
    app.run_server(debug=True)
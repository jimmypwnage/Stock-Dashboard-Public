import pandas as pd
import numpy as np
import scipy.optimize as sco
import datetime


def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    return std, returns


# minimize is used because scipy does not have maximise, so we minimise instead minimize the negative
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # setting variable bounds
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[0]

def portfolio_return(weights, mean_returns, cov_matrix):
    return -portfolio_annualized_performance(weights, mean_returns, cov_matrix)[1]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def max_return(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_return, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result


if __name__ == '__main__':
    interested_df = pd.read_csv('interested_df.csv')
    interested_df['weight'] = interested_df['value_invested'] / interested_df['value_invested'].sum()

    stock_data = pd.read_csv('stock_data.csv')
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    end_date_default = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date_default = (datetime.datetime.today() - datetime.timedelta(weeks=52)).strftime('%Y-%m-%d')

    table = pd.pivot_table(data=stock_data, index='date', columns='symbol', values='adj_close', aggfunc='sum')
    returns = table.pct_change()

    ### CONSTANTS ###
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 25000
    risk_free_rate = 0.01252  # T-Bills yield singapore
    num_stocks = stock_data['symbol'].nunique()

    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252

    min_vol = min_variance(mean_returns, cov_matrix)
    # get min volatility weights
    sdp_min, rp_min = portfolio_annualized_performance(min_vol['x'], mean_returns, cov_matrix)

    # get max sharpe ratio
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

    sdp_max, rp_max = portfolio_annualized_performance(max_sharpe['x'], mean_returns, cov_matrix)

    target = np.linspace(rp_min, rp_max, 100)

    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)

    # p fun is the volatility x-axis, target is the targetted return. i.e. calculate the best volatility for the target interested
    weights = [p['x'] for p in efficient_portfolios]
    volatility = [p['fun'] for p in efficient_portfolios]

    efficient_df = pd.DataFrame({'weight': weights,
                                 'volatility': volatility,
                                 'returns': target
                                 })
    efficient_df[mean_returns.index.tolist()] = pd.DataFrame(efficient_df['weight'].tolist(), index=efficient_df.index)

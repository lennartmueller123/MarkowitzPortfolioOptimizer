from dash import Dash, dcc, html, Input, Output, dash_table
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
from datetime import datetime, timedelta
import quantstats as qs
import plotly.graph_objs as go

# Initialize Dash app
app = Dash(__name__)

# Time horizon options
time_horizons = {
    '3 Months': 3,
    '1 Year': 12,
    '3 Years': 36,
    '5 Years': 60,
    '10 Years': 120
}
# Risk preference options
risk_preferences = ['Max_Sharpe', 'Min_Volatility', 'Specific_Risk']

# App layout
app.layout = html.Div([
    html.H1("Markowitz Portfolio Optimizer", style={'textAlign': 'center', 'font-size': '50px'}),

    # Time horizon dropdown
    html.Label("Select Investment Time Horizon:",
               style={'display': 'block',
                      'textAlign': 'center'}),
    dcc.Dropdown(
        id='time-horizon-dropdown',
        options=[{'label': period, 'value': period} for period in time_horizons.keys()],
        value='',  # Default value
        style={'width': '50%', 'margin': 'auto'}
    ),

    # Risk preference dropdown
    html.Label("Select Investment Risk Preferences:",
               style={'display': 'block',
                      'textAlign': 'center'}),
    dcc.Dropdown(
        id='risk-preference-dropdown',
        options=[{'label': pref, 'value': pref} for pref in risk_preferences],
        value='Select...',  # Default value
        style={'width': '50%', 'margin': 'auto'}  # 'margin-left': '0', 'margin-right': 'auto'
    ),

    # Container for target volatility input
    html.Div(id='target-volatility-container', children=[
        html.Label("Target Volatility:",
                   style={'display': 'block', 'textAlign': 'center'}),
        dcc.Input(
            id='target-volatility-input',
            type='number', value=0.1, min=0, max=1, step=0.01,
            style={'display': 'block', 'margin': 'auto'}
        )
    ]),

    # Button to trigger optimization
    html.Div(
        html.Button('Optimize Portfolio', id='optimize-button', n_clicks=0),
        style={'textAlign': 'center', 'margin': 'auto'}
    ),

    # Display optimized portfolio
    html.Div(id='optimized-portfolio'),

    # Add the performance metrics to the layout
    html.Div(id='performance-metrics')
], style={'backgroundColor': 'black', 'color': '#39FF14'})

# Add a Graph component to your layout for the correlation matrix
app.layout.children.append(dcc.Graph(id='return-plot'))


# Callback to show/hide target volatility input
@app.callback(
    Output('target-volatility-container', 'style'),
    [Input('risk-preference-dropdown', 'value')]
)
def toggle_target_volatility_input(risk_preference):
    if risk_preference == 'Specific_Risk':
        return {'display': 'block'}  # Show input
    else:
        return {'display': 'none'}  # Hide input


# Callback for optimization and metrics
@app.callback(
    [Output('optimized-portfolio', 'children'),
     Output('performance-metrics', 'children'),
     Output('return-plot', 'figure')],
    [Input('optimize-button', 'n_clicks')],
    [Input('time-horizon-dropdown', 'value'),
     Input('risk-preference-dropdown', 'value'),
     Input('target-volatility-input', 'value')]
)
def update_output(n_clicks, time_horizon, risk_preference, target_volatility):
    if n_clicks > 0:
        # Calculate start and end dates
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        months = time_horizons[time_horizon]
        start_date = end_date - timedelta(days=30 * months)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        # Fetch and process data
        tickers = "AXP AMGN AAPL BA CAT CSCO CVX GS HD HON IBM INTC JNJ KO JPM MCD MMM MRK MSFT NKE PG TRV UNH CRM VZ V WBA WMT DIS"
        data = yf.download(tickers, start=start_date, end=end_date, group_by='tickers')
        prices = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers.split()})

        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        # Optimize portfolio based on risk preference
        ef = EfficientFrontier(mu, S)
        try:
            if risk_preference == 'Max_Sharpe':
                weights = ef.max_sharpe()
            elif risk_preference == 'Min_Volatility':
                weights = ef.min_volatility()
            elif risk_preference == 'Specific_Risk':
                weights = ef.efficient_risk(target_volatility)
            else:
                return (html.Div("Invalid risk preference selected."), html.Div())

            cleaned_weights = ef.clean_weights()
            # Filtering out all zero weights and sort rest in descending order
            filtered_sorted_weights = {k: v for k, v in
                                       sorted(cleaned_weights.items(), key=lambda item: item[1], reverse=True)
                                       if v > 0}

            # Display optimized weights
            optimized_portfolio_div = html.Div([
                html.H3("Optimized Portfolio Weights:"),
                html.Pre(str(filtered_sorted_weights))
            ], style={'display': 'block',
                      'textAlign': 'center'})

            # Calculate daily returns for the optimized portfolio
            daily_returns = prices.pct_change().dropna()
            optimized_portfolio_returns = (daily_returns * pd.Series(cleaned_weights)).sum(axis=1)

            # Fetch Dow Jones Index data and calculate its daily returns
            dow_jones_data = yf.download('^DJI', start=start_date, end=end_date)
            dow_jones_returns = dow_jones_data['Adj Close'].pct_change().dropna()

            # Convert to QuantStats format
            optimized_portfolio_returns.index = pd.to_datetime(optimized_portfolio_returns.index)
            dow_jones_returns.index = pd.to_datetime(dow_jones_returns.index)

            # Calculate the performance metrics
            sharpe_ratio = qs.stats.sharpe(optimized_portfolio_returns)
            sharpe_ratio_benchmark = qs.stats.sharpe(dow_jones_returns)
            annual_return = qs.stats.comp(optimized_portfolio_returns)
            annual_return_benchmark = qs.stats.comp(dow_jones_returns)
            volatility = qs.stats.volatility(optimized_portfolio_returns)
            volatility_benchmark = qs.stats.volatility(dow_jones_returns)
            consecutive_wins = qs.stats.consecutive_wins(optimized_portfolio_returns)
            consecutive_wins_benchmark = qs.stats.consecutive_wins(dow_jones_returns)
            consecutive_losses = qs.stats.consecutive_losses(optimized_portfolio_returns)
            consecutive_losses_benchmark = qs.stats.consecutive_losses(dow_jones_returns)
            cagr = qs.stats.cagr(optimized_portfolio_returns)
            cagr_benchmark = qs.stats.cagr(dow_jones_returns)

            # Calculate cumulative returns
            cumulative_returns_portfolio = (1 + optimized_portfolio_returns).cumprod() - 1

            # Fetch Dow Jones Index data and calculate its cumulative returns
            dow_jones_data = yf.download('^DJI', start=start_date, end=end_date)
            dow_jones_returns = dow_jones_data['Adj Close'].pct_change()
            cumulative_returns_dow = (1 + dow_jones_returns).cumprod() - 1

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns_portfolio.index, y=cumulative_returns_portfolio, mode='lines',
                                     name='Optimized Portfolio'))
            fig.add_trace(go.Scatter(x=cumulative_returns_dow.index, y=cumulative_returns_dow, mode='lines',
                                     name='Dow Jones Index'))
            fig.update_layout(title='Portfolio Performance vs Dow Jones Index',
                              title_x=0.5,
                              xaxis_title='Date',
                              yaxis_title='Cumulative Returns',
                              plot_bgcolor='black',
                              paper_bgcolor='black',
                              font=dict(color='#39FF14'),
                              title_font=dict(color='#39FF14', size=20),
                              xaxis=dict(
                                  titlefont=dict(color='#39FF14'),
                                  tickfont=dict(color='#39FF14'),
                                  gridcolor='#5e5e5e'  # Adjust grid color for visibility
                              ),
                              yaxis=dict(
                                  titlefont=dict(color='#39FF14'),
                                  tickfont=dict(color='#39FF14'),
                                  gridcolor='#5e5e5e'  # Adjust grid color for visibility
                              ),
                              legend=dict(
                                  font=dict(color='#39FF14')
                              )
                              )

            # Prepare performance metrics data for the DataTable
            metrics_data = [
                {"Metric": "Annual Return", "Strategy": f"{annual_return:.2f}",
                 "Benchmark": f"{annual_return_benchmark:.2f}"},
                {"Metric": "Sharpe Ratio", "Strategy": f"{sharpe_ratio:.2f}",
                 "Benchmark": f"{sharpe_ratio_benchmark:.2f}"},
                {"Metric": "Volatility", "Strategy": f"{volatility:.2f}", "Benchmark": f"{volatility_benchmark:.2f}"},
                {"Metric": "Consecutive Wins", "Strategy": f"{consecutive_wins:.2f}",
                 "Benchmark": f"{consecutive_wins_benchmark:.2f}"},
                {"Metric": "Consecutive Losses", "Strategy": f"{consecutive_losses:.2f}",
                 "Benchmark": f"{consecutive_losses_benchmark:.2f}"},
                {"Metric": "CAGR", "Strategy": f"{cagr:.2f}", "Benchmark": f"{cagr_benchmark:.2f}"}
            ]

            # Create the DataTable for the Performance Metrics
            metrics_table = dash_table.DataTable(
                data=metrics_data,
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': 'Strategy', 'id': 'Strategy'},
                    {'name': 'Benchmark', 'id': 'Benchmark'}
                ],
                style_cell={'textAlign': 'left',
                            'backgroundColor': 'black'},
                style_header={
                    'backgroundColor': 'black',
                    'fontWeight': 'bold',
                    'font-size': '18px'
                }
            )

            # Wrap the table in a container to set the width
            metrics_table_container = html.Div(
                children=[metrics_table],
                style={'width': '50%', 'margin': 'auto'}
            )

            return optimized_portfolio_div, metrics_table_container, fig

        except ValueError as e:
            min_volatility = ef.min_volatility()
            min_volatility_value = round(min(min_volatility.values()), 4)
            error_message = html.Div([
                html.H3("Error: Specified target volatility is not achievable."),
                html.P(f"The minimum achievable volatility is: {min_volatility_value}")
            ])
            return html.Div(), error_message

    return html.Div(), html.Div(), go.Figure()


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

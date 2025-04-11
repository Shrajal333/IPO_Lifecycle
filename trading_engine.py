import os
import tqdm
import random
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def try_parse_date(date_str):
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")

def average_timestamp(timestamp1, timestamp2):

    if isinstance(timestamp1, str):
        timestamp1 = datetime.fromisoformat(timestamp1)
    if isinstance(timestamp2, str):
        timestamp2 = datetime.fromisoformat(timestamp2)

    avg_timestamp = timestamp1 + (timestamp2 - timestamp1) / 2
    return avg_timestamp

def price_loader(ma_type, ma_period, excel_file, end_date, interval):

    companies = pd.read_excel(excel_file) # IPOs where QIB > 20x
    tickers = companies["Company Ticker"].tolist() # Loading company tickers
    prices_dict = {}

    for ticker in tqdm.tqdm(tickers):
        prices_df = yf.download(ticker, end=end_date, interval=interval, progress=False)[["Close", "High", "Low"]] # Downloading weekly closing and high prices

        prices_df.to_csv('prices.csv')
        prices_df = pd.read_csv("prices.csv", skiprows=3, names=["Date", "Close", "High", "Low"])

        if ma_type == "SMA":
            prices_df[str(ma_period) + " " + ma_type] = prices_df['Close'].rolling(window=ma_period).mean().fillna(0) # Adding SMA to the dataframe
        else:
            prices_df[str(ma_period) + " " + ma_type] = prices_df['Close'].ewm(span=ma_period).mean().fillna(0) # Adding SMA to the dataframe

        prices_dict[ticker] = prices_df.set_index("Date") # Seting index to datetime
        os.remove("prices.csv")

    return prices_dict

def trade_statistics(trades_random, profits):

    positive_profits = [p for p in profits if p > 0]
    negative_profits = [p for p in profits if p < 0]

    total_trades = len(profits) # Storing total number of trades
    avg_profit = sum(positive_profits) / len(positive_profits) if positive_profits else 0 # Storing average profit percentages
    avg_loss = sum(negative_profits) / len(negative_profits) if negative_profits else 0 # Storing average loss percentages
    max_profit = max(positive_profits, default=0) # Storing max profit percentage
    max_loss = min(negative_profits, default=0) # Storing max loss percentage
    trade_win_percentage = (len(positive_profits) / len(profits)) * 100 if profits else 0 # Storing trade win perentage

    risk_reward_ratio = (-avg_profit / avg_loss) if avg_loss != 0 else 0 # Storing risk-reward ratio
    profit_factor = (avg_profit * trade_win_percentage / 100) / ((1 - trade_win_percentage / 100) * -avg_loss) if avg_loss != 0 and trade_win_percentage != 100 else 0 # Storing profit factor

    overall_periods = []
    profitable_periods = []
    losing_periods = []

    for _, start_pos, end_pos, start_date, end_date in trades_random: # Parse the dates
        try:
            start_date = try_parse_date(start_date)
            end_date = try_parse_date(end_date)
        except:
            pass

        holding_period = (end_date - start_date).days # Calculate holding period in days
        overall_periods.append(holding_period)

        if end_pos > start_pos:
            profitable_periods.append(holding_period)
        else:
            losing_periods.append(holding_period)

    # Calculate averages
    avg_overall = sum(overall_periods) / len(overall_periods)
    avg_profitable = sum(profitable_periods) / len(profitable_periods) if profitable_periods else 0
    avg_losing = sum(losing_periods) / len(losing_periods) if losing_periods else 0

    return total_trades, avg_profit, avg_loss, max_profit, max_loss, trade_win_percentage, risk_reward_ratio, profit_factor, avg_overall, avg_profitable, avg_losing

def trade_visualization(profits):

    sizes = [abs(p) * 10 for p in profits]  # Storing sizes according to magnitude of percentages
    colors = ['green' if p > 0 else 'red' for p in profits]  # Green color for profit and red color for losses

    ax = plt.subplot(1, 2, 1)  # Subplot 1
    ax.scatter(range(len(profits)), profits, c=colors, s=sizes, alpha=0.6, edgecolors="w", linewidth=0.5)
    ax.set_title('Profit and Loss of Trades', fontsize=16)
    ax.set_xlabel('Trade Number', fontsize=12)
    ax.set_ylabel('Profit/Loss %', fontsize=12)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Profit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Loss')
    ], loc='upper left')

def upper_lower_curves(trades, initial_portfolio, portfolio_size=10):
    trade_data = pd.DataFrame(trades, columns=["ticker", "start", "end", "start_date", "end_date"])
    trade_data["profits"] = (initial_portfolio / portfolio_size) * trade_data["end"] / trade_data["start"] - (initial_portfolio / portfolio_size)

    avg_std = np.std(trade_data["profits"])
    avg_profit = np.mean(trade_data["profits"])
    trade_number = np.arange(0, initial_portfolio * 10 + 1)

    curve_upper = trade_number * avg_profit + np.sqrt(trade_number) * 3 * avg_std
    curve_expected = trade_number * avg_profit * np.random.uniform(0.9, 1.0, len(trade_number))
    curve_lower = trade_number * avg_profit - np.sqrt(trade_number) * 3 * avg_std

    ax = plt.subplot(1, 2, 2)  # Subplot 2
    ax.plot(trade_number, curve_upper, label="Upper Curve", color="green")
    ax.plot(trade_number, curve_expected, label="Expected Curve", color="black")
    ax.plot(trade_number, curve_lower, label="Lower Curve", color="red")
    ax.set_xlabel("Number of Trades")
    ax.set_ylabel("Profit")
    ax.set_title("Upper and Lower Profit Curves")
    ax.legend()
    ax.grid(True)

def monte_carlo_simulation(trades, randomizer_simulation, initial_portfolio, position_size, num_simulations):

    portfolio_values = []
    max_drawdowns = []
    worst_drawdown_curve = []  # To store equity curve for worst drawdown
    worst_drawdown = 0  # Initialize the worst drawdown value

    for _ in range(num_simulations):

        trades_random = random.sample(trades, int(np.round(len(trades), 0) * randomizer_simulation))
        profits = [(exit_price - entry_price) / entry_price * 100 for _, entry_price, exit_price, _, _ in trades_random]
        profits = [profit for profit in profits if profit < 400] # Removing profit percentage more than 400% (exceptional profits)
        np.random.shuffle(profits) # Shuffle trades randomly

        # Simulate portfolio with position sizing
        num_trades = len(profits)
        portfolio = initial_portfolio
        cumulative_values = [portfolio]
        max_drawdown = 0
        peak = portfolio  # Initialize peak value

        for i in range(num_trades):
            # Scale trade impact based on position size
            trade_allocation = portfolio / position_size
            portfolio += (trade_allocation * profits[i]) / 100
            cumulative_values.append(portfolio)

            # Update peak and calculate drawdown
            if portfolio > peak:
                peak = portfolio
            drawdown = (peak - portfolio) / peak
            max_drawdown = max(max_drawdown, drawdown)

        portfolio_values.append(portfolio)
        max_drawdowns.append(max_drawdown)

        if max_drawdown > worst_drawdown:
            worst_drawdown = max_drawdown
            worst_drawdown_curve = cumulative_values # Storing the curve for maximum drawdown simulation

    all_portfilio_returns = [((value / initial_portfolio) ** (1 / 5) - 1) * 100 for value in portfolio_values]
    avg_final_return = np.median(all_portfilio_returns) / 100

    max_drawdown = np.median(max_drawdowns)
    max_std = np.std(profits)

    sharpe_ratio = avg_final_return * 100 / max_std
    calmar_ratio = avg_final_return / max_drawdown

    # Plot the results
    _, axes = plt.subplots(1, 3, figsize=(18, 3), constrained_layout=True)

    # Equity curve during the worst drawdown
    axes[0].plot(worst_drawdown_curve, label=(
                                            f"Position Size: {position_size} stocks\n"
                                            f"Percent trades taken: {randomizer_simulation: .2%}\n"
                                            f"Median Annual Return: {avg_final_return:.2%}\n"
                                            f"Median Max Drawdown: {max_drawdown:.2%}\n"
                                            f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                                            f"Calmar Ratio: {calmar_ratio:.2f}"))
    axes[0].set_title("Equity Curve During Maximum Drawdown")
    axes[0].set_xlabel("Trade Number")
    axes[0].set_ylabel("Portfolio Value")
    axes[0].legend()
    axes[0].grid(True)

    # Profit percentage distribution
    axes[1].hist(all_portfilio_returns, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title("Profit Percentage Distribution")
    axes[1].set_xlabel("Profit (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)
    axes[1].axvline(np.mean(all_portfilio_returns), color='black', linestyle='-', linewidth=3, label=f'Mean: {np.mean(all_portfilio_returns):.2f}%')
    axes[1].axvline(np.percentile(all_portfilio_returns, 5), color='black', linestyle='-', linewidth=3, label=f'5th Percentile: {np.percentile(all_portfilio_returns, 10):.2f}%')
    axes[1].axvline(np.percentile(all_portfilio_returns, 95), color='black', linestyle='-', linewidth=3, label=f'95th Percentile: {np.percentile(all_portfilio_returns, 90):.2f}%')
    axes[1].legend()

    # Drawdown distribution
    axes[2].hist(max_drawdowns, bins=30, color='red', edgecolor='black', alpha=0.7)
    axes[2].set_title("Drawdown Percentage Distribution")
    axes[2].set_xlabel("Drawdown (%)")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True)
    axes[2].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axes[2].axvline(np.mean(max_drawdowns), color='black', linestyle='-', linewidth=3, label=f'Mean: {100 * np.mean(max_drawdowns):.2f}%')
    axes[2].axvline(np.percentile(max_drawdowns, 5), color='black', linestyle='-', linewidth=3, label=f'5th Percentile: {100 * np.percentile(max_drawdowns, 10):.2f}%')
    axes[2].axvline(np.percentile(max_drawdowns, 95), color='black', linestyle='-', linewidth=3, label=f'95th Percentile: {100 * np.percentile(max_drawdowns, 90):.2f}%')
    axes[2].legend()

    axes[1].set_xlim(0, 100)
    axes[2].set_xlim(0, 0.3)
    plt.show()

def metric_loader(trades, initial_portfolio, position_sizing, num_simulations, randomizer_overall):

    trades_random = random.sample(trades, int(np.round(len(trades), 0) * randomizer_overall))
    profits = [(exit_price - entry_price) / entry_price * 100 for _, entry_price, exit_price, _, _ in trades_random] # Storing profit percentage calculated from entry and exit
    total_trades, avg_profit, avg_loss, max_profit, max_loss, trade_win_percentage, risk_reward_ratio, profit_factor, avg_overall, avg_profitable, avg_losing = trade_statistics(trades_random, profits)

    metrics = {
            "Total Trades": total_trades,
            "Average Profit %": avg_profit,
            "Average Loss %": avg_loss,
            "Max Profit %": max_profit,
            "Max Loss %": max_loss,
            "Average Holding days": avg_overall,
            "Average Holding Profitable days": avg_profitable,
            "Average Holding Losing days": avg_losing,
            "Trade Win %": trade_win_percentage,
            "Risk-Reward Ratio": risk_reward_ratio,
            "Profit Factor": profit_factor,
        }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    print(metrics_df)

    plt.figure(figsize=(24, 3))
    trade_visualization(profits)
    upper_lower_curves(trades_random, initial_portfolio)

    for size in position_sizing:
        randomizer_simulation = min(((360 * size * 5 / avg_overall) / total_trades), 0.99)
        monte_carlo_simulation(trades, randomizer_simulation, initial_portfolio, size, num_simulations)
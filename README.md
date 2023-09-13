# AutoTrade: Stock Trading Automation with ML and API Trading

## Overview

This project aims to fully automate stock trading by integrating machine learning models with the Alpaca API for trade execution. Designed to function as a comprehensive trading system, it is built on a modular architecture that includes distinct components for data downloading, stock price prediction, and trading strategy implementation.

### Machine Learning Models
The system employs specialized machine learning models for predictive analytics. These models, which can be configured for either regression or classification tasks, are designed to understand and forecast stock price movements. Additionally, they can dynamically retrain based on new data and are highly customizable to align with various trading strategies.

### Alpaca API Integration
The Alpaca API is integral to the project, serving as the execution engine for trades. It enables the system to place market orders, manage portfolios, and acquire real-time or historical market data across a variety of stock tickers.

### Trading Indicators and Strategies
The project leverages a plethora of trading indicators such as RSI, MACD, and EMA to assist the machine learning models in making more precise buy/sell decisions. These indicators are configurable, providing flexibility to adapt to different trading strategies.

### GitHub Actions for Automation
Thanks to GitHub Actions, the entire trading workflow—from data acquisition and price prediction to order execution—is automated. The system is set to operate from Monday to Friday at 3:30 PM ET, ensuring a consistent trading schedule.

By synergizing these elements, the project delivers a robust, adaptable, and fully automated stock trading system.

**Note: This project is intended for educational purposes and should not be used as financial advice.**

## Prerequisites

- Python 3.x
- yfinance==0.2.28
- pandas==2.1.0
- numpy==1.23.5
- tqdm==4.66.1
- scikit-learn==1.2.2
- pytz==2023.3
- finta==1.3
- ta==0.10.2
- keras==2.13.1
- tensorflow==2.13.0
- alpaca-py==0.10.0
- matplotlib==3.7.2
- tqdm==4.66.1
- scipy==1.11.2

## Installation

First, clone the repository:

\`\`\`bash
git clone https://github.com/yourusername/yourrepository.git
\`\`\`

Navigate to the project directory:

\`\`\`bash
cd your repository
\`\`\`

Install the required Python packages:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Modules

1. `data_downloader`: Responsible for downloading stock data.
2. `price_prediction`: Uses machine learning models to predict stock prices.
3. `trading_strategy`: Determines trading decisions based on price predictions and other indicators.
4. `models`: contains a set of machine learning models to predict stock prices

## Configuration Parameters

The script provides a set of command-line arguments to customize its behavior. Below are important parameters with a detailed explanation for each.

### Timing Intervals

- `--start_date`: Specifies the start date for the data. Format should be a string (e.g., "2020-01-01"). Default is `None`.
- `--days`: Number of days for which data is needed. Default is `350`.
- `--room_na`: Number of days for padding missing data. Default is `200`.
- `--interval`: The time interval for data. Choices are `1d`, `1wk`, `1mo`. Default is `1d`.
- `--end_date`: Specifies the end date for the data. Default is today's date.

### Stock Data

- `--stock`: Ticker symbol of the stock to consider for data download. Default is `AAPL`.

### Log Indicators

- `--need_log`: Flag to determine if log transformation is needed. Default is `False`.

### Technical Indicators

- `--need_ta`: Flag to check if technical indicators like RSI, STOCH are needed. Default is `False`.
- `--choices_ta`: List of technical indicators to consider. Default is `['RSI','STOCH']`.

### MACD Indicators

- `--need_macd`: Flag to check if MACD indicators are needed. Default is `False`.
- `--columns_macd`: Target column for MACD computation. Default is `['Close']`.
- `--short_span`: Short span period for MACD. Default is `12`.
- `--long_span`: Long span period for MACD. Default is `26`.
- `--signal_span`: Signal span for MACD. Default is `9`.

### EMA Indicators

- `--need_ema`: Flag to check if EMA indicators are needed. Default is `False`.
- `--days_ema`: Days for which EMA is calculated. Default is `[5,15,25,50]`.
- `--columns_ema`: Target column for EMA computation. Default is `['Close']`.

### Previous Day Indicators

- `--need_prev`: Flag to check if previous day indicators are needed. Default is `False`.
- `--columns_prev`: Target column for previous day indicators. Default is `['Close']`.
- `--days_prev`: Number of previous days to consider. Default is `20`.

### Backtesting

- `--shift_target`: Flag to check if shifting the target for backtesting is needed. Default is `False`.

### Machine Learning Settings

- `--regression`: Flag for regression or classification tasks. Default is `False`.
- `--training_size`: Number of days used for training the model. Default is `300`.
- `--training_interval`: Interval to train a new model for each testing day. Default is `1`.

### Data Preprocessing

- `--return_normalization`: Flag to enable return rate normalization. Default is `True`.
- `--min_max_normalization`: Flag to enable Min-Max normalization. Default is `True`.
- `--standard_normalization`: Flag to enable standard normalization. Default is `False`.

### API Keys for Alpaca

- `--api_id`: Your API ID for the Alpaca service.
- `--api_secret_key`: Your API secret key for the Alpaca service.

Example usage:

\`\`\`bash
python execution.py --start_date=2022-01-01 --days=350 --stock=AAPL
\`\`\`

## GitHub Actions Workflow

### Automated Trading Schedule

This project uses GitHub Actions to automate the trading script. The workflow is scheduled to run at 3:30 PM ET (8:30 PM UTC) from Monday to Friday. After customizing your trading strategy and hyperparamters in execution.py, Github Actions automates the trading process based on predetermined schedule on each trading day.

### Workflow Configuration

The `.github/workflows` directory contains the YAML configuration file for the workflow.



## API Trading Functions

- `GetPosition(trading_client)`: Retrieves the current position in the given stock.
- `PlaceOrderBUY(trading_client, ticker, quantity)`: Places a buy order.
- `PlaceOrderSELL(trading_client, ticker, quantity)`: Places a sell order.

**Important:**  Please register your own Alpaca API ID and Secret key for API trading services

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

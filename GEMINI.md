# Project Overview

This project is an algorithmic trading system designed for Binance Futures, specifically for BTCUSDT. It employs a sophisticated, multi-faceted approach to automate trading decisions:

1.  **Data Acquisition & Processing:** Fetches historical K-line data from Binance and stores it in an SQLite database for efficient access and analysis.
2.  **Feature Engineering & Indicator Calculation:** Utilizes the `pandas_ta` library to calculate a wide array of technical indicators (e.g., Exponential Moving Averages (EMA), Relative Strength Index (RSI), Average True Range (ATR), Average Directional Index (ADX)) which serve as inputs for the trading strategy and regime classification.
3.  **Market Regime Classification:** A core component of the system is a PyTorch-based Transformer model (`TradingTransformerV2`). This deep learning model is trained using `src/train_model.py` on preprocessed data to classify the current market into different "regimes" (e.g., trending, ranging, volatile). This classification informs the trading strategy.
4.  **Trading Strategy & Execution:**
    *   **Primary Signal:** An EMA crossover strategy is used to generate initial buy/sell signals.
    *   **Pre-Entry Filters:** Before a trade is executed, several critical filters are applied to enhance robustness and risk management:
        *   **Regime Filter:** Ensures that the market is in a "favorable" regime (as determined by the Transformer model and ADX/ATR% thresholds) for trade entry.
        *   **Temporal Filter:** Restricts trading to specific, predefined hours of the day and days of the week to avoid unfavorable market conditions (e.g., low liquidity periods).
        *   **Circuit Breaker:** Implements a risk-control mechanism that prevents new trade entries after a predefined number of consecutive losses, enforcing a cooldown period to protect capital.
    *   **Position Sizing:** Calculates optimal position size based on a predefined risk-per-trade percentage and the calculated stop-loss distance.
    *   **Order Management:** Executes market orders for entry and automatically places corresponding stop-loss (SL) and take-profit (TP) orders.
    *   **Position Management:** Actively manages open positions, including hybrid liquidation protocols that incorporate time-based stops and dynamic ATR trailing stops to lock in profits or limit losses.
5.  **Operational Infrastructure:**
    *   **Configuration:** Sensitive information such as API keys and email credentials are securely loaded from a `.env` file.
    *   **Logging & Notifications:** Comprehensive logging is implemented for monitoring bot activity, and email notifications are sent for critical events (e.g., errors, position entries/exits).
    *   **State Persistence:** The bot's current position state (side, quantity, entry price, SL/TP) is persisted to a JSON file to ensure continuity across restarts.
    *   **Scheduling:** The `schedule` library is used to run the main trading logic periodically (e.g., hourly).

# Main Technologies

*   **Python:** The primary programming language for the entire system.
*   **Binance API (`python-binance`):** Used for real-time interaction with the Binance Futures exchange (fetching data, placing orders, managing positions).
*   **Pandas & NumPy:** Essential libraries for data manipulation, analysis, and numerical operations on K-line data and indicators.
*   **Pandas-TA:** A powerful library for calculating a wide range of technical analysis indicators.
*   **PyTorch:** The deep learning framework used to build, train, and deploy the Transformer-based market regime classification model.
*   **Scikit-learn (`joblib`):** Used for saving and loading machine learning artifacts, specifically the label encoder for regime classification.
*   **SQLite:** Utilized as a lightweight database to store historical K-line data.
*   **`python-dotenv`:** For managing environment variables and sensitive credentials.
*   **`schedule`:** A job scheduling library used to automate the periodic execution of the trading logic.
*   **`smtplib`:** Python's built-in library for sending email notifications.

# Building and Running

## Dependencies

This project relies on several Python libraries. While a `requirements.txt` file was not found, the following key libraries are essential and should be installed in your Python environment:

*   `python-binance`
*   `pandas`
*   `numpy`
*   `pandas_ta`
*   `torch` (PyTorch)
*   `scikit-learn`
*   `python-dotenv`
*   `schedule`
*   `tqdm`
*   `seaborn`
*   `matplotlib`

You can typically install these using pip:
```bash
pip install python-binance pandas numpy pandas_ta torch scikit-learn python-dotenv schedule tqdm seaborn matplotlib
```

## Configuration

1.  **Environment Variables:** Create a `.env` file in the project's root directory (`C:\Monilusion\`) and populate it with your Binance API keys and email notification settings. An example `.env` file structure would be:

    ```
    BINANCE_API_KEY=YOUR_BINANCE_API_KEY
    BINANCE_SECRET_KEY=YOUR_BINANCE_SECRET_KEY
    EMAIL_SENDER_ADDRESS=your_email@example.com
    EMAIL_SENDER_PASSWORD=your_email_password # Or app-specific password
    EMAIL_RECEIVER_ADDRESS=recipient_email@example.com
    SMTP_SERVER=smtp.example.com
    SMTP_PORT=587
    ```

    **Security Note:** Be extremely cautious with your API keys and passwords. Never commit your `.env` file to version control.

## Running the Trading Bot

The main trading bot logic is contained within `real_M1.py`. It is designed to run continuously, scheduling its core trading job periodically.

To start the trading bot:

```bash
python C:\Monilusion\real_M1.py
```

The bot will log its activities to `live_trading_bot_phase1.log` and send email notifications for critical events if configured.

## Training the Regime Classification Model

The Transformer-based market regime classification model can be trained independently. This requires preprocessed data to be available in the `data/processed_data_regime_final` directory.

To train the model:

```bash
python C:\Monilusion\src\train_model.py
```

The trained model will be saved to the `saved_models_regime_transformer_v2` directory.

# Development Conventions

*   **Modular Structure:** The project follows a modular design, with core components organized within the `src` directory. This includes subdirectories for `filters`, `policy`, `position`, `exits`, and main modules like `data_pipeline`, `feature_engineer`, `model`, `regime_labeler`, `train_model`, and `utils`.
*   **Legacy Code:** Older or experimental scripts are maintained in the `legacy` directory.
*   **Configuration Management:** Trading parameters are centrally managed within the `TRADING_PARAMS` dictionary in `real_M1.py`, allowing for easy adjustment of strategy parameters.
*   **Logging:** Extensive logging is used throughout the application to provide detailed insights into the bot's operations, aid in debugging, and monitor performance.
*   **Environment Variables:** Sensitive credentials are handled via environment variables loaded from a `.env` file, promoting secure practices.
*   **Data Storage:** Historical K-line data is stored in an SQLite database, providing a persistent and efficient data source.

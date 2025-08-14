# Algorithmic Trading Bot for Binance Futures

This project implements a sophisticated algorithmic trading system designed for automated trading on Binance Futures, specifically for the BTCUSDT pair. It integrates advanced machine learning techniques with robust trading strategies and comprehensive risk management.

## Features

*   **Automated Trading:** Executes trades automatically based on predefined strategies and real-time market data.
*   **Market Regime Classification:** Utilizes a PyTorch-based Transformer model to classify market conditions (e.g., trending, ranging, volatile), enabling adaptive trading decisions.
*   **Dynamic Strategy & Filters:** Employs an EMA crossover strategy enhanced with multiple pre-entry filters (regime, temporal, circuit breaker) for improved robustness.
*   **Advanced Risk Management:** Includes intelligent position sizing, automated Stop-Loss (SL) and Take-Profit (TP) orders, and hybrid liquidation protocols (time-based stops, ATR trailing stops).
*   **Data Management:** Fetches and stores historical K-line data in an SQLite database.
*   **Operational Monitoring:** Features extensive logging and email notification for critical events.

## Technologies Used

*   **Python:** The core programming language.
*   **Binance API (`python-binance`):** For real-time exchange interaction.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Pandas-TA:** For technical analysis indicator calculations.
*   **PyTorch:** For deep learning model development (Market Regime Classifier).
*   **SQLite:** For historical data storage.
*   **`python-dotenv`:** For secure environment variable management.
*   **`schedule`:** For automated task scheduling.

## Getting Started

### Prerequisites

Ensure you have Python 3.x and `pip` installed.

### Installation

Install the necessary Python libraries using pip:

```bash
pip install python-binance pandas numpy pandas_ta torch scikit-learn python-dotenv schedule tqdm seaborn matplotlib
```

### Configuration

1.  **Create a `.env` file:** In the root directory of the project (`C:\Monilusion\`), create a file named `.env`.
2.  **Add your credentials:** Populate the `.env` file with your Binance API keys and email notification settings. **Do NOT commit this file to version control.**

    ```
    BINANCE_API_KEY=YOUR_BINANCE_API_KEY
    BINANCE_SECRET_KEY=YOUR_BINANCE_SECRET_KEY
    EMAIL_SENDER_ADDRESS=your_email@example.com
    EMAIL_SENDER_PASSWORD=your_email_password # Or app-specific password
    EMAIL_RECEIVER_ADDRESS=recipient_email@example.com
    SMTP_SERVER=smtp.example.com
    SMTP_PORT=587
    ```

## Usage

### Running the Trading Bot

The main trading bot logic is in `real_M1.py`. It is designed to run continuously, scheduling its core trading job periodically (e.g., hourly).

To start the trading bot:

```bash
python real_M1.py
```

The bot will log its activities to `live_trading_bot_phase1.log` and send email notifications for critical events if configured.

### Training the Regime Classification Model

The Transformer-based market regime classification model can be trained independently. This requires preprocessed data to be available in the `data/processed_data_regime_final` directory.

To train the model:

```bash
python src/train_model.py
```

The trained model will be saved to the `saved_models_regime_transformer_v2` directory.

## Project Structure

*   `src/`: Contains the core modules of the trading system (data pipeline, feature engineering, models, filters, etc.).
*   `data/`: Stores historical market data and preprocessed datasets.
*   `saved_models/`: Location for trained machine learning models.
*   `legacy/`: Contains older or experimental scripts.
*   `cudare.py`: A GPU-accelerated backtesting script for strategy optimization.
*   `real_M1.py`: The live trading bot implementation.

## License

(Optional: Add your license information here)

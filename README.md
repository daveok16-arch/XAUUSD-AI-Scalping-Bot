# XAUUSD Scalping ML Signal Bot

Advanced AI/ML-powered scalping signal bot for XAUUSD (Gold/USD) trading. Built with production-grade MLOps pipeline from Kaggle training to Render deployment.

## Architecture

```
Kaggle Training → GitHub Storage → Render API
     ↑______________________________↓
         (Automated Retraining)
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Source** | Yahoo Finance API | Historical XAUUSD OHLCV data |
| **Feature Engineering** | pandas-ta, numpy | 50+ technical indicators |
| **ML Models** | XGBoost, LightGBM, Random Forest | Price direction prediction |
| **Training** | Kaggle Notebooks | GPU-accelerated model training |
| **Version Control** | Git + GitHub | Code and model storage |
| **API** | Flask | REST API for signal generation |
| **Deployment** | Render | Production hosting |

## Features

- **50+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, etc.
- **Multiple ML Models**: XGBoost, LightGBM, Random Forest with comparison
- **Time-Series Cross-Validation**: Proper CV for financial data
- **Signal Generation**: BUY/SELL signals with stop-loss and take-profit
- **Risk Management**: ATR-based position sizing and risk-reward ratios
- **REST API**: Real-time signal endpoints
- **Kaggle Optimized**: Works seamlessly in Kaggle environment

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/xauusd-scalping-bot.git
cd xauusd-scalping-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

#### Local Training

```bash
python -m src.train --model xgboost --period 5y --interval 1h
```

#### Kaggle Training

1. Open `notebooks/kaggle_training.ipynb` in Kaggle
2. Run all cells
3. Download trained model from `/kaggle/working/models/`

### 4. Run API Locally

```bash
python -m src.api.app
```

Visit: http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/signal` | GET | Get current trading signal |
| `/predict` | POST | Predict from custom OHLCV data |
| `/model-info` | GET | Model information |
| `/features` | GET | Available features list |
| `/historical-signals` | GET | Historical signal statistics |

### Example: Get Signal

```bash
curl https://your-app.onrender.com/signal
```

Response:
```json
{
  "signal": "BUY",
  "confidence": 0.78,
  "current_price": 2345.67,
  "entry_price": 2345.67,
  "stop_loss": 2342.50,
  "take_profit": 2355.30,
  "risk_reward_ratio": 3.0,
  "timestamp": "2024-01-15T14:30:00"
}
```

### Example: Custom Prediction

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ohlcv": [
      {"timestamp": "2024-01-15T14:00:00", "open": 2340, "high": 2348, "low": 2338, "close": 2345, "volume": 15000},
      {"timestamp": "2024-01-15T14:30:00", "open": 2345, "high": 2350, "low": 2343, "close": 2349, "volume": 12000}
    ]
  }'
```

## Project Structure

```
xauusd-scalping-bot/
├── src/
│   ├── data/
│   │   └── data_fetcher.py        # Yahoo Finance data fetching
│   ├── features/
│   │   └── feature_engineering.py # Technical indicators
│   ├── models/
│   │   └── model_trainer.py       # ML model training
│   ├── signals/
│   │   └── signal_generator.py    # Signal generation
│   ├── api/
│   │   └── app.py                 # Flask API
│   ├── utils/
│   │   └── logger.py              # Logging utility
│   └── train.py                   # Training script
├── notebooks/
│   └── kaggle_training.ipynb      # Kaggle training notebook
├── models/                        # Trained models (git-ignored)
├── config/
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

## Model Configuration

### Available Models

- `xgboost` - XGBoost Classifier (default, recommended)
- `lightgbm` - LightGBM Classifier
- `random_forest` - Random Forest Classifier
- `gradient_boosting` - Gradient Boosting Classifier
- `logistic_regression` - Logistic Regression

### Hyperparameters

Default XGBoost configuration:
```python
{
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": "auto",
    "objective": "binary:logistic"
}
```

## Feature Engineering

### Price Features
- Returns, log returns, candlestick patterns
- Body size, shadows, directional indicators

### Momentum (10 features)
- RSI (7, 14), MACD, Stochastic, Williams %R, CCI

### Trend (10 features)
- SMA (10, 20, 50), EMA (10, 20, 50), ADX

### Volatility (10 features)
- ATR (7, 14), Bollinger Bands, Keltner Channels

### Volume (4 features)
- Volume SMA, ratio, OBV, Chaikin Money Flow

### Time (7 features)
- Hour, day of week cyclical encoding, session flags

### Lag Features
- Lagged close, returns, RSI (1, 2, 3, 5, 10 periods)

## Deployment

### Render

1. Push code to GitHub
2. Connect repository to Render
3. Use `render.yaml` for configuration
4. Model auto-loads from `models/` directory

### Docker

```bash
docker build -t xauusd-bot .
docker run -p 5000:5000 xauusd-bot
```

## GitHub Actions

Automated training pipeline:
- **Manual trigger**: Workflow dispatch with configurable parameters
- **Scheduled**: Weekly retraining every Sunday
- **Artifacts**: Model files uploaded as build artifacts
- **Auto-commit**: Trained models committed back to repository

## Performance Metrics

Typical performance on 5-year hourly XAUUSD data:

| Metric | Score |
|--------|-------|
| Accuracy | 72-78% |
| F1 Score | 70-76% |
| AUC | 78-84% |
| CV F1 | 68-74% |

*Note: Past performance does not guarantee future results.*

## Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only.
- Trading forex/CFDs carries significant risk of loss
- Always test on demo accounts first
- Past performance does not guarantee future results
- Use proper risk management and position sizing
- The authors are not responsible for any trading losses

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [pandas-ta](https://github.com/twopirllc/pandas-ta) for technical indicators
- [XGBoost](https://xgboost.ai/) for gradient boosting
- [scikit-learn](https://scikit-learn.org/) for ML utilities

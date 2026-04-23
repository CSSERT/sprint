<h1 align="center">
  <img src="./assets/logo.svg" width="300" height="75" /><br />
  <em><sub><sup>Stock Price Intelligent Tool</sup></sub></em>
</h1>

<!----------------------------------------------------------------------------->

A set of tools for forecasting and anomaly detection on stock prices.

## :minidisc: Installation

Requirements:

1. Python >= 3.14
2. uv >= 0.11

Install all dependencies:

```bash
uv sync --all-packages
```

## :toolbox: Usage

For general use, start the FastAPI server:

```bash
uv run poe start
```

During development, start the FastAPI server in `dev` mode:

```bash
uv run poe dev
```

## :gear: Examples

`/v1/forecast` request:

```json
{
  "ticker": "VCB",
  "interval": "daily",
  "lookback_days": 1
}
```

`/v1/forecast` response:

```json
{
  "ticker": "VCB",
  "interval": "daily",
  "predictions": [
    {
      "date": "2025-11-03",
      "step": 1,
      "quantiles": {
        "0.1": 30.8246,
        "0.5": 40.0799,
        "0.9": 57.4146
      }
    },
    {
      "date": "2025-11-07",
      "step": 5,
      "quantiles": {
        "0.1": 32.279,
        "0.5": 42.9604,
        "0.9": 53.5482
      }
    },
    {
      "date": "2025-11-14",
      "step": 10,
      "quantiles": {
        "0.1": 30.7611,
        "0.5": 35.5878,
        "0.9": 59.7114
      }
    },
    {
      "date": "2025-11-28",
      "step": 20,
      "quantiles": {
        "0.1": 32.5877,
        "0.5": 36.8659,
        "0.9": 59.8161
      }
    }
  ],
  "history": [
    {
      "date": "2025-10-31",
      "close": 59.6
    }
  ]
}
```

# Executive Summary

Three classes of time series were defined: 

- Low data: less than 5 datapoints
- Zero-inflated: more than 5 datapoints, more than 40% zeros
- Full series: all other cases

Three models were considered for forecasting:

- Naive (historical mean and std)
- TSB (for zero-inflated series)
- ARIMA

## 1. Input data

- Distinct time series (`INVENTORY_ID` x `REGION`): **368**
- Minimum date: **2025-07-13**
- Maximum date: **2025-12-21**

Series count by class:

| Series class | Number of series |
|---|---:|
| Low data | 132 |
| Zero-inflated | 67 |
| Full series | 169 |

## 2. Model selection

Current notebook selection criterion:

- Error type: **point**
- Selection metric: **MAE**
- Optimization direction: **minimize**

Relative frequency of best-model selection by class:

| Series class | Best model | Relative frequency |
|---|---|---:|
| Low data | naive | 100.00% |
| Zero-inflated | tsb | 53.66% |
| Zero-inflated | naive | 46.34% |
| Full series | arima | 100.00% |

## 3. Error summary

Average selected-model error by class (based on current criterion = MAE):

| Series class | Average MAE |
|---|---:|
| Low data | 10.480 |
| Zero-inflated | 7.166 |
| Full series | 9.537 |

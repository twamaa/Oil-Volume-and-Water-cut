# Oil Production Volume and Water Cut Prediction  
**Classical Models vs. TCN-Transformer Hybrid Model**  

## Project Overview
This project addresses the challenge of forecasting **oil production volume** and **water cut** using both classical statistical models and advanced deep learning models. The goal is to compare their performance on real-world oilfield datasets and assess how well each handles the complex nonlinearities and dependencies in the data.

The project consists of two phases:  
- **Replication Phase:** Validating results from the literature using ARIMA, Prophet, and LSTM on a univariate dataset.  
- **Extended Phase:** Applying models to a multivariate dataset with additional features, introducing Bidirectional LSTM and a **TCN-Transformer Hybrid Model** for improved performance.

---

## Objectives
- Predict daily oil production volume and water cut rates.  
- Compare classical time series models (ARIMA, Prophet) with deep learning models (LSTM, Bidirectional LSTM, and TCN-Transformer).  
- Evaluate model performance using RMSE and MAE to understand accuracy, robustness, and limitations.  

---

## Dataset Description

### Replication Phase Dataset  
- **Source:** Historical daily oil production data from 4 wells (CSV format).  
- **Features:**  
  - Date  
  - Oil production volume  

### Extended Phase Dataset  
- **Source:** Enhanced real-world oilfield dataset.  
- **Features:**  
  - Date  
  - Oil production volume  
  - Water cut (%)  
  - Gas volume  
  - Reservoir pressure  
  - Dynamic level  
  - Working hours  

---

## Methodology

### Replication Phase
- **Data Preprocessing:**  
  - Merged multiple CSV files.  
  - Formatted into time-indexed univariate data.  
  - Applied stationarity tests (ADF) and differencing (d=1).  
- **Models Applied:**  
  - **ARIMA (1,1,1):** Linear, good at trend detection.  
  - **Prophet:** Captures trend and seasonality, but responds poorly to sudden changes.  
  - **Basic LSTM:** Single-layer, underfitted without deeper architecture.  

### Extended Phase
- **Data Preprocessing:**  
  - Multivariate data scaling and time windowing for deep learning models.  
  - Differencing for ARIMA and Prophet to ensure stationarity.  
- **Models Applied:**  
  - **ARIMA:** Classical statistical forecasting.  
  - **Prophet:** Handles trend and seasonality on univariate data.  
  - **LSTM:**  
    - Basic: One hidden layer (64 units).  
    - Deep LSTM: Added extra layers and dropout (0.2) for regularization.  
  - **Bidirectional LSTM:**  
    - Processes data both forward and backward in time for better long-term dependency modeling.  
  - **TCN-Transformer Hybrid:**  
    - **TCN Block:** Captures local temporal patterns with dilated convolutions.  
    - **Transformer Encoder:** Learns global dependencies via attention mechanisms.  
    - **Positional Encoding (PE):** Further improves interpretability and accuracy.  

---

## Model Performance

| Model                        | Dataset        | RMSE (Oil) | MAE (Water) | Remarks                                        |
|------------------------------|----------------|------------|-------------|------------------------------------------------|
| ARIMA (1,1,1)                | Replication    | 1.06       | 0.82        | Captures trend, misses sudden drops            |
| Prophet                      | Replication    | Not listed | Not listed  | Good on seasonality, poor on sudden changes    |
| Basic LSTM                   | Replication    | ~1.1       | ~1.0        | Underfitting observed                          |
| ARIMA                        | Extended       | Not listed | Not listed  | Applied on multivariate data                   |
| Prophet                      | Extended       | Not listed | Not listed  | Applied on univariate oil production           |
| Basic LSTM                   | Extended       | ~1.1       | ~1.0        | Nonlinear modeling, but struggles with water cut |
| Deep LSTM                    | Extended       | ↓ Slight   | ↓ Slight    | Improved generalization                        |
| Bidirectional LSTM           | Extended       | Best LSTM  | Best LSTM   | Best among LSTM models                         |
| **TCN-Transformer (w/o PE)** | Extended       | **1.0997** | **0.9460**  | Robust, better than LSTM variants              |
| **TCN-Transformer (with PE)**| Extended       | **1.0958** | **0.9264**  | Best overall; attention boosts accuracy        |

>  **Observation:** TCN-Transformer with positional encoding outperforms all other models in both oil and water cut forecasting.

---

## Conclusion
- **Classical models** like ARIMA and Prophet perform well for linear trends and simple patterns but struggle with sudden shifts and nonlinear dependencies.  
- **LSTM models** improve performance with deeper architectures, but Bidirectional LSTM clearly outperforms vanilla LSTM for long-range dependencies.  
- **The TCN-Transformer hybrid model** achieved the best performance, demonstrating robustness in capturing both local (via TCN) and global (via Transformer) temporal patterns.

---

## Limitations
- The dataset size limits the generalizability of deep learning models.  
- External factors like equipment downtime or weather weren’t included in the modeling.  
- Hyperparameter tuning was limited due to computational constraints.  

---

## Future Work
- Integrate more exogenous variables (e.g., weather, operational logs).  
- Explore ensemble methods combining classical and deep learning models.  
- Develop long-horizon forecasts with uncertainty quantification.  
- Deploy the best model as an API for real-time monitoring and prediction.  

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Data Handling: `Pandas`, `NumPy`  
  - Classical Models: `Statsmodels` (ARIMA), `Prophet` (Meta/Facebook Prophet)  
  - Deep Learning: `TensorFlow` / `Keras`  
  - Visualization: `Matplotlib`, `Seaborn`  
  - Misc: `scikit-learn` for preprocessing  

---



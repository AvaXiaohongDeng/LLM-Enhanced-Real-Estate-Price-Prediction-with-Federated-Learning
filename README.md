# LLM-Enhanced Real Estate Price Prediction with Federated Learning

This repository contains the code for a thesis project on real estate price prediction using both structured and unstructured data. The framework combines traditional machine learning models, LLM-based feature extraction, federated learning, and agentic workflows to support price prediction, privacy-preserving training, listing optimization, and personalized recommendations.

## Project Overview

This project presents a hybrid machine learning framework for real estate price prediction.

- Structured data such as property attributes and macroeconomic indicators is processed using traditional machine learning methods.
- Unstructured data such as listing descriptions is analyzed using large language models to extract additional predictive features.
- Federated learning is integrated to support privacy-preserving model training across distributed datasets.
- Agent-based workflows are designed for both sellers and buyers, including price suggestions, listing optimization, and preference-based recommendations.


## Main Components

### Structured Data Modeling

The structured-data pipeline focuses on preprocessing, feature selection, and baseline model evaluation using real estate and macroeconomic data.

### Unstructured Data + LLMs

The unstructured-data pipeline uses listing text and large language models to extract useful features that enhance prediction performance.

### Federated Learning

The federated learning modules support distributed training for Random Forest and LSTM-based workflows while improving data privacy.

### Agentic Workflows

The project includes seller-side and buyer-side agentic workflows for listing optimization, price feedback, preference matching, and recommendation support.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AvaXiaohongDeng/LLM-Enhanced-Real-Estate-Price-Prediction-with-Federated-Learning.git
cd LLM-Enhanced-Real-Estate-Price-Prediction-with-Federated-Learning
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If different submodules use separate dependency files, install them from their corresponding folders as needed.

### 4. Run notebooks or federated modules

- Run the notebooks in sequence for structured-data experiments.
- Run scripts in `llm/` for LLM-based feature extraction and agentic workflows.
- Run scripts in `fl_flower_rf/` and `fl_flower_lstm/` for federated learning experiments.

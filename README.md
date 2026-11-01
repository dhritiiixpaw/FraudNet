# 🛡️ FraudNet
### Privacy-Aware Fraud Detection & Investigation Prototype

FraudNet is a hackathon project exploring how simulated financial institutions could identify suspicious transactions and account connections while minimizing the sharing of raw transaction data.

The prototype combines machine learning, graph-based investigation signals, and an interactive dashboard to support fraud analysis using synthetic financial transaction data.

## ✨ Features

- **Machine Learning:** Decision Tree and Random Forest models for transaction-level fraud classification.
- **Graph-Based Investigation:** Explore account connections associated with known fraud labels in the dataset.
- **Investigation Reports:** Generate sample reports from model-flagged transactions.
- **Interactive Dashboard:** View transaction statistics, model alerts, graph-linked transactions, and investigation reports.
- **AI-Assisted Investigation:** Generate investigation summaries using Gemini AI.
- **Privacy-Aware Concept:** Explore a simulated multi-bank scenario in which limited fraud signals could be shared instead of raw transaction data.

## 🧰 Tech Stack

- Python
- Pandas
- Scikit-learn
- NetworkX
- Streamlit
- Google Gemini API
- PaySim synthetic financial transaction dataset

## 📊 Dataset

FraudNet uses the [PaySim synthetic financial transaction dataset](https://www.kaggle.com/datasets/ealaxi/paysim1).

The full dataset contains approximately 6.36 million simulated transactions. The current prototype uses the first 500,000 rows for model training, evaluation, and dashboard demonstrations.

**Dataset characteristics:**
- Synthetic financial transactions
- Includes transaction types, amounts, account balances, and fraud labels
- Fraud labels are provided by the dataset

This project does not use real banking customer data.

## 🤖 Model Performance

The Random Forest model was evaluated on a held-out test set containing 100,000 transactions.

| Metric | Result |
|---|---:|
| Accuracy | 99.98% |
| Fraud Precision | 92.59% |
| Fraud Recall | 53.19% |
| Fraud cases correctly flagged | 25 of 47 |
| False positives | 2 |
| False negatives | 22 |

Although overall accuracy is high, the dataset is highly imbalanced. Fraud recall is therefore an important metric: the model identified approximately 53% of the fraud cases in the test set.

These results are from a prototype evaluated on synthetic data and should not be interpreted as real-world banking performance.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd FraudNet
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train and evaluate the models

```bash
python src/train_model.py
```

This trains the Decision Tree and Random Forest models and displays their evaluation results.

### 4. Run graph-based investigation

```bash
python src/graph_detection.py
```

This generates graph-linked investigation signals based on known fraud labels in the dataset.

### 5. Generate sample investigation reports

```bash
python src/generate_reports.py
```

### 6. Launch the dashboard

```bash
python -m streamlit run src/dashboard.py
```

Open the local URL displayed in the terminal to access the FraudNet dashboard.

### Gemini AI Configuration

The AI-assisted investigation feature uses the Google Gemini API.

Create a `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_api_key_here
```

Keep your API key private. Do not commit the `.env` file to GitHub.

## ⚠️ Limitations

- PaySim is a synthetic dataset and does not represent real banking transactions.
- The current prototype does not implement real cross-bank infrastructure or secure privacy-preserving computation.
- Graph-linked investigation signals use known fraud labels as seeds and are not independent graph-model predictions.
- Model alerts are predictions and do not confirm that a transaction is fraudulent.
- Generated investigation reports are samples and are not official regulatory filings.
- Model evaluation is limited to the first 500,000 rows and has not been validated on real-world financial data.

## 🔮 Future Improvements

- Simulated multi-bank transaction environments
- Independent graph-based suspicious activity detection
- Evaluation on a larger portion of the dataset
- Privacy-preserving fraud signal sharing
- Enhanced investigation report generation
- AI-assisted explanations grounded in model findings

## 👩‍💻 Project

**FraudNet — Hackathon Prototype**

Built using Python, machine learning, graph analysis, and Streamlit.


# fraud-detection-system

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-F7931E?style=for-the-badge&logo=scikit-learn)

> **A real-time machine learning system that detects fraudulent credit card transactions by analyzing geospatial, temporal, and behavioral patterns.**

---

## Live Demo

![System Demo](media/dashboard_demo.gif)

*(Note: If the video player above does not load, you can [download the demo here](media/dashboard_demo.mp4).)*

---

## The Problem
Banks lose billions annually to credit card fraud. Traditional rule-based systems often miss subtle patterns or generate too many false alarms. This project solves this by using a **Random Forest Classifier** to learn complex, non-linear relationships between transaction features.

**Key Challenges Addressed:**
* **Class Imbalance:** Fraud makes up <0.5% of transactions. We handled this using **Class Weights** and **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure high recall.
* **Context Awareness:** A ₹5,000 transaction is normal at 5 PM but suspicious at 3 AM. A transaction in Delhi is normal, but one in Paris 10 minutes later from same person is very suspicious.

---

## Key Features

### 1. Geospatial Geofencing (PyDeck)
* **The Logic:** We calculate the **Haversine Distance** (as-the-crow-flies) between the user's registered home address (New Delhi) and the merchant's location.
* **The Visual:** A dynamic "Safe Zone" radius is drawn on the map. Transactions falling far outside this radius trigger high-risk alerts.

### 2. Localized for India
* Adapted to handle **Indian Rupee (₹)** inputs.
* Calibrated for Indian geographic coordinates.
* Integration of dark-mode maps to focus on data signals rather than political borders.

### 3. Explainable AI (XAI)
* Instead of a black-box "Yes/No", the system provides a **Probabilistic Risk Score** (0-100%).
* **Gauge Charts** (Plotly) visualize the confidence level, helping analysts prioritize investigations.

---

## Technical Architecture

### Tech Stack
| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Model** | Random Forest | Robust classification algorithm resistant to overfitting. |
| **Data Processing** | Pandas & NumPy | Feature engineering (Distance, Age, Hour extraction). |
| **Persistence** | Joblib | Serializing the trained model for instant inference. |
| **Frontend** | Streamlit | Rapid prototyping of the web dashboard. |
| **Visualization** | PyDeck & Plotly | Interactive 3D maps and responsive charts. |

### Feature Engineering Pipeline
The raw dataset contained basic fields. We engineered the following **High-Value Features**:
1.  `distance_km`: Calculated from Lat/Long coordinates.
2.  `hour_of_day`: Extracted from timestamps (Fraud spikes 2 AM - 4 AM).
3.  `age`: Derived from Date of Birth.
4.  `category_enc`: Label Encoded merchant categories.

---

## Installation & Setup

### Prerequisites
* Python 3.8+
* pip

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

```

### Step 2: Generate the Model

1. Open the training notebook: `train_model.ipynb`
2. Run all cells to train the Random Forest on your machine.
3. This will automatically create the `model/` folder containing `fraud_model.pkl`.

### Step 3: Run the Dashboard

Once the model is generated, launch the application:

```bash
streamlit run app.py

```

---

## Project Structure

```text
fraud-detection-system/
├── data/                   # Raw CSV files
├── media/                  # Demo videos and assets
│   └── dashboard_demo.mp4 
├── model/                  # Serialized Model Artifacts (Generated locally)
│   ├── fraud_model.pkl     # The trained Random Forest
│   ├── features.pkl        # Column names for consistency
│   └── category_encoder.pkl# LabelEncoder for categorical data
├── app.py                  # Streamlit Dashboard Application
├── train_model.ipynb       # Feature Engineering & Training Lab
└── README.md               # Project Documentation

```

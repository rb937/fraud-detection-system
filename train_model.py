import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Currency mapping: the raw dataset is in USD, but the app operates in INR.
# Keep the model trained on USD and convert in the app (1 USD = 95 INR).
USD_TO_INR = 95.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


print("Loading data...")
df = pd.read_csv('data/fraudTrain.csv', nrows=500000)

print("Engineering features...")
df['distance_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

df['dob'] = pd.to_datetime(df['dob'])
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

df['hour'] = df['trans_date_trans_time'].dt.hour

features = ['category', 'amt', 'gender', 'age', 'distance_km', 'hour']
X = df[features].copy()
y = df['is_fraud']

X['gender'] = X['gender'].map({'M': 0, 'F': 1})

le_cat = LabelEncoder()
X['category'] = le_cat.fit_transform(X['category'])

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Training Model (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

print("\n--- Model Performance on Test Set ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Sample Accuracy:", model.score(X_test, y_test))

print("Saving model...")
joblib.dump(model, 'model/fraud_model.pkl')
joblib.dump(le_cat, 'model/category_encoder.pkl')
joblib.dump(X.columns, 'model/features.pkl')

print("DONE! Model saved to 'model/' folder.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 1. Load dataset
df = pd.read_csv("data.csv")

# Drop customerID column if present
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Handle missing values
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])


# 2.Feature-target split

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 3. Define base models

clf1 = GradientBoostingClassifier(random_state=42)
clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf3 = AdaBoostClassifier(random_state=42)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)],
    voting='soft'
)


# 4. Train model
voting_clf.fit(X_train, y_train)


# 5. Evaluate
y_pred = voting_clf.predict(X_test)

print("=== Voting Classifier Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 6. Save model

joblib.dump(voting_clf, "churn_model.pkl")
print("\nModel saved as churn_model.pkl")

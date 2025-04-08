import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (use your own path for the dataset)
df = pd.read_csv("/content/synthetic_intrusion_dataset.csv")

# Drop unnecessary columns
df.drop(columns=["timestamp", "source_ip", "destination_ip"], inplace=True)

# Encode categorical variables
le_protocol = LabelEncoder()
df["protocol"] = le_protocol.fit_transform(df["protocol"])

le_attack = LabelEncoder()
df["attack_category"] = le_attack.fit_transform(df["attack_category"])

# Split features and target
X = df.drop(columns=["attack_category"])
y = df["attack_category"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

# Train and evaluate each classifier
accuracy_results = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[name] = accuracy
    report = classification_report(y_test, y_pred, output_dict=True)

    precision_scores[name] = report['weighted avg']['precision']
    recall_scores[name] = report['weighted avg']['recall']
    f1_scores[name] = report['weighted avg']['f1-score']

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}\n")

# Plot accuracy results
plt.figure(figsize=(10, 5))
plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Comparison of Classifier Accuracies")
plt.xticks(rotation=45)
plt.show()

# Plot Precision, Recall, F1-score for each classifier
metrics_df = pd.DataFrame({
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores
})

metrics_df.plot(kind='bar', figsize=(10, 5))
plt.xlabel("Classifier")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-Score for Each Classifier")
plt.xticks(rotation=45)
plt.legend()
plt.show()

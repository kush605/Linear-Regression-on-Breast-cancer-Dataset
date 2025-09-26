# Import necessary libraries
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical computations
import matplotlib.pyplot as plt     # For plotting graphs
import seaborn as sns               # For enhanced data visualization
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.preprocessing import StandardScaler     # To standardize features
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc  # Evaluation metrics

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("breast_cancer_dataset.csv")

# Display first 5 rows to understand dataset structure
print("First 5 rows of dataset:")
print(data.head())

# Display dataset info: column names, data types, missing values
print("\nDataset Info:")
print(data.info())

# ==============================
# 2. Define Features & Target
# ==============================
# Separate independent variables (features) and dependent variable (target)
X = data.drop("target", axis=1)   # All columns except the target column
y = data["target"]                # Target column (0 = negative, 1 = positive)

# ==============================
# 3. Train-Test Split
# ==============================
# Split data into training and testing sets (80% train, 20% test)
# stratify=y ensures that the class distribution is maintained in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 4. Standardize Features
# ==============================
# Standardization helps models converge faster and improves performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit scaler on train data & transform
X_test_scaled = scaler.transform(X_test)         # Transform test data using same scaler

# ==============================
# 5. Train Logistic Regression
# ==============================
model = LogisticRegression(max_iter=1000)  # Initialize logistic regression (allow more iterations)
model.fit(X_train_scaled, y_train)          # Train model on scaled training data

# Predictions on test set
y_pred = model.predict(X_test_scaled)          # Class predictions
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability estimates for positive class

# ==============================
# 6. Evaluation
# ==============================
# Print detailed classification metrics (precision, recall, f1-score, support)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # Annotate cells with numbers
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve (Receiver Operating Characteristic)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)  # Compute false positive & true positive rates
roc_auc = auc(fpr, tpr)                           # Calculate Area Under Curve (AUC)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color="blue")  # ROC curve
plt.plot([0,1], [0,1], color="red", linestyle="--")                        # Diagonal line (random guess)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# ==============================
# 7. Threshold Tuning Example
# ==============================
# By default, logistic regression uses threshold=0.5
# You can change threshold to balance precision & recall
threshold = 0.3   # Example threshold (can try 0.4, 0.5, 0.6, etc.)
y_pred_custom = (y_prob >= threshold).astype(int)  # Convert probabilities to 0/1 based on threshold

# Evaluate model with custom threshold
print(f"\nConfusion Matrix at threshold {threshold}:")
print(confusion_matrix(y_test, y_pred_custom))
print(f"\nClassification Report at threshold {threshold}:\n", classification_report(y_test, y_pred_custom))

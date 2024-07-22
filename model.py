import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Step 2: Preprocess the data
# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical labels
df['smoking_history'] = df['smoking_history'].map({'never': 0, 'former': 1, 'current': 2, 'ever': 3, 'not current': 4, 'No Info': 5})

# Convert 'gender' to numerical (assuming binary classification)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Separate features (X) and target variable (y)
X = df.drop(['diabetes'], axis=1)
y = df['diabetes']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train the classifier
rf_classifier.fit(X_train, y_train)

# Step 6: Evaluate the model (Optional)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Step 7: Save the trained model to a file in the current directory
joblib.dump(rf_classifier, 'random_forest_model.pkl')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Load and Clean the Data
file_path = 'ec2dataset_2.csv'
data = pd.read_csv(file_path)

cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']
for column in cost_columns:
    data[column] = pd.to_numeric(data[column].astype(str).str.replace(r'[$, hourly]', '', regex=True), errors='coerce')

# Step 3: Feature Engineering
data['Instance Memory'] = pd.to_numeric(data['Instance Memory'].astype(str).str.replace(' GiB', '', regex=False), errors='coerce')
data['vCPUs'] = pd.to_numeric(data['vCPUs'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')

# Step 4: Handle Missing Data
data_cleaned = data.dropna(subset=['On Demand', 'Instance Memory', 'vCPUs'])
print(f"Total valid rows for training: {len(data_cleaned)}")

# Step 5: Split the Data
X = data_cleaned[['Instance Memory', 'vCPUs']]
y = data_cleaned['On Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Step 6: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\nIntercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Step 8: Visualize the Results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted On-Demand Costs')
plt.xlabel('Actual On-Demand Cost ($)')
plt.ylabel('Predicted On-Demand Cost ($)')
plt.show()

# Step 9: Make Predictions
# Supplying a DataFrame with matching feature names prevents UserWarnings during prediction
new_instance = pd.DataFrame([[4, 2]], columns=['Instance Memory', 'vCPUs'])
predicted_cost = model.predict(new_instance)
print(f"\nPredicted On-Demand Cost for 4 GiB, 2 vCPUs: ${predicted_cost[0]:.4f}")
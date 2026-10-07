import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="EC2 Cost Analysis & Prediction")
st.title("EC2 Cost Analysis & Prediction")

# Load and clean data globally
@st.cache_data
def load_data():
    data = pd.read_csv('ec2dataset.csv')
    cost_cols = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']
    for col in cost_cols:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[$, hourly]', '', regex=True), errors='coerce')
    
    # Feature engineering for regression
    data['Instance Memory Numeric'] = pd.to_numeric(data['Instance Memory'].astype(str).str.replace(' GiB', '', regex=False), errors='coerce')
    data['vCPUs Numeric'] = pd.to_numeric(data['vCPUs'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    return data

data = load_data()
cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']

# --- PART 1: Cost Analysis ---
st.header("Part 1: Cost Analysis")

st.subheader("Summary Statistics")
st.dataframe(data[cost_columns].describe())

st.subheader("Cost Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data[cost_columns], palette="Set2", ax=ax1)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

st.subheader("T2 vs T3 Family Comparison")
t2 = data[data['Name'].str.startswith('t2', na=False) | data['Name'].str.startswith('T2', na=False)].assign(Family='T2')
t3 = data[data['Name'].str.startswith('t3', na=False) | data['Name'].str.startswith('T3', na=False)].assign(Family='T3')
comparison = pd.concat([t2, t3])

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Family', y='On Demand', data=comparison, palette="muted", ax=ax2)
st.pyplot(fig2)

# --- PART 2: Cost Prediction (Linear Regression) ---
st.header("Part 2: Predicting Costs")

# Prepare data
data_cleaned = data.dropna(subset=['On Demand', 'Instance Memory Numeric', 'vCPUs Numeric'])
X = data_cleaned[['Instance Memory Numeric', 'vCPUs Numeric']]
y = data_cleaned['On Demand']

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Absolute Error (MAE)", f"${mean_absolute_error(y_test, y_pred):.4f}")
col2.metric("Mean Squared Error (MSE)", f"${mean_squared_error(y_test, y_pred):.4f}")
col3.metric("Root Mean Squared Error (RMSE)", f"${mean_squared_error(y_test, y_pred)**0.5:.4f}")

# Visualize Actual vs Predicted
st.subheader("Actual vs Predicted On-Demand Costs")
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(y_test, y_pred, alpha=0.7, color='b')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
ax3.set_xlabel('Actual On-Demand Cost ($)')
ax3.set_ylabel('Predicted On-Demand Cost ($)')
st.pyplot(fig3)

# Interactive Prediction
st.subheader("Test a Custom Configuration")
test_mem = st.number_input("Memory (GiB)", min_value=0.5, value=4.0)
test_cpu = st.number_input("vCPUs", min_value=1, value=2)

if st.button("Predict Cost"):
    pred = model.predict(pd.DataFrame([[test_mem, test_cpu]], columns=['Instance Memory Numeric', 'vCPUs Numeric']))
    final_cost = max(0, pred[0]) # Forces negative numbers to 0
    st.success(f"Predicted On-Demand Cost: **${final_cost:.4f} / hour**")
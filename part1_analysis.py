import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
file_path = 'ec2dataset_2.csv'
data = pd.read_csv(file_path)

print("Dataset Info:")
print(data.info())

# Step 3: Clean the Data
cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']
for column in cost_columns:
    # Converting to string first prevents errors with NaN values
    data[column] = pd.to_numeric(data[column].astype(str).str.replace(r'[$, hourly]', '', regex=True), errors='coerce')

print("\nMissing values after conversion:")
print(data[cost_columns].isnull().sum())

# Step 4: Perform Summary Analysis
cost_summary = data[cost_columns].describe()
print("\nCost Summary Statistics:")
print(cost_summary)

# Step 5: Visualize the Data
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[cost_columns], palette="Set2")
plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()

# Step 6: Identify Outliers
def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

outliers_on_demand = detect_outliers('On Demand')
print(f"\nNumber of On-Demand Outliers Found: {len(outliers_on_demand)}")

# Step 7 & 8: Filter and Compare Different Instance Families
# AWS instance names are typically lowercase in the raw data (e.g., 't2', 't3')
def filter_instance_family(family):
    return data[data['Name'].str.startswith(family, na=False)]

t2_instances = filter_instance_family('t2')
t3_instances = filter_instance_family('t3')

print("\nT2 Instance Costs Summary:\n", t2_instances[cost_columns].describe())
print("\nT3 Instance Costs Summary:\n", t3_instances[cost_columns].describe())

# Visualize cost comparison for T2 vs T3 instance families side-by-side
comparison = pd.concat([
    t2_instances[['Name', 'On Demand', 'Linux Reserved cost']].assign(Family='T2'),
    t3_instances[['Name', 'On Demand', 'Linux Reserved cost']].assign(Family='T3')
])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Family', y='On Demand', data=comparison, palette="muted")
plt.title('On-Demand Cost Distribution: T2 vs T3 Instances', fontsize=16)
plt.ylabel('Cost (USD)', fontsize=12)
plt.show()
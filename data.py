import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
np.random.seed(42)

# Customers
customers = pd.DataFrame({
    "customer_id": range(1001, 1101),
    "name": [fake.name() for _ in range(100)],
    "age": np.random.randint(18, 80, 100),
    "income": np.random.uniform(30000, 200000, 100),
    "location": [fake.city() for _ in range(100)]
})
customers.to_csv("customers.csv", index=False)

# Transactions
transactions = pd.DataFrame({
    "transaction_id": range(2001, 2201),
    "customer_id": np.random.choice(customers["customer_id"], 200),
    "date": [fake.date_between(start_date="-1y", end_date="today") for _ in range(200)],
    "type": np.random.choice(["Deposit", "Withdrawal", "Payment"], 200),
    "amount": np.random.uniform(100, 10000, 200)
})
transactions.to_csv("transactions.csv", index=False)

# Credit Data
credit_data = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "credit_score": np.random.uniform(300, 850, 100),
    "loan_id": range(3001, 3101),
    "loan_amount": np.random.uniform(5000, 100000, 100),
    "loan_duration": np.random.randint(12, 60, 100),  # 1-5 years
    "current_balance": lambda x: x["loan_amount"] * np.random.uniform(0.2, 0.9, 100),  # 20-90% remaining
    "loan_status": np.random.choice(["Active", "Defaulted", "Paid"], 100),
    "credit_history": np.random.randint(1, 20, 100)
})
credit_data["current_balance"] = credit_data.apply(
    lambda row: row["loan_amount"] * np.random.uniform(0.2, 0.9), axis=1
)
credit_data.to_csv("credit_data.csv", index=False)

# Collateral
collateral = pd.DataFrame({
    "customer_id": np.random.choice(customers["customer_id"], 50),
    "collateral_type": np.random.choice(["Vehicle", "Property"], 50),
    "collateral_value": np.random.uniform(10000, 200000, 50),
    "loan_id": range(3001, 3051)
})
collateral.to_csv("collateral.csv", index=False)

# Deposits
deposits = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "account_id": range(4001, 4101),
    "balance": np.random.uniform(1000, 50000, 100),
    "account_type": np.random.choice(["Savings", "Checking", "Fixed"], 100)
})
deposits.to_csv("deposits.csv", index=False)

# Behavioral Data
behavior = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "feedback_score": np.random.uniform(0, 5, 100),
    "dispute_count": np.random.randint(0, 5, 100)
})
behavior.to_csv("behavior.csv", index=False)

print("Synthetic data generated: customers.csv, transactions.csv, credit_data.csv, collateral.csv, deposits.csv, behavior.csv")
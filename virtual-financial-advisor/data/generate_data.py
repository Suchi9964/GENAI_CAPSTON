import pandas as pd
import numpy as np

np.random.seed(2025)

num_rows = 5200
user_ids = [f"user_{i}" for i in range(1, 21)]
categories_income = ['Salary', 'Bonus', 'Interest']
categories_expense = ['Groceries', 'Rent', 'Utilities', 'Entertainment', 'Dining', 'Transport', 'Healthcare', 'Education', 'Savings Transfer']

payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Transfer']
merchants = ['Amazon', 'Walmart', 'Netflix', 'Uber', 'Starbucks', 'Local Grocery', 'Electric Company', 'Landlord', 'Hospital', 'School']

dates = pd.date_range('2023-01-01', '2024-12-31').to_pydatetime().tolist()

data = []
for _ in range(num_rows):
    user = np.random.choice(user_ids)
    date = np.random.choice(dates)
    is_income = np.random.choice([True, False], p=[0.15, 0.85])
    if is_income:
        category = np.random.choice(categories_income)
        amount = round(np.random.uniform(1000, 5000), 2)
    else:
        category = np.random.choice(categories_expense)
        amount = round(-np.random.uniform(5, 500), 2)
    payment_method = np.random.choice(payment_methods)
    merchant = np.random.choice(merchants)
    description = f"{category} payment at {merchant}"
    transaction_id = f"txn_{np.random.randint(1000000, 9999999)}"

    data.append([transaction_id, user, date, category, amount, payment_method, merchant, description])

df = pd.DataFrame(data, columns=[
    'transaction_id', 'user_id', 'date', 'category', 'amount',
    'payment_method', 'merchant', 'description'
])

df.to_csv('virtual_financial_advisor_data.csv', index=False)
print("CSV file with 5200 rows created successfully.")

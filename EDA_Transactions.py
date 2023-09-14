import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# Read the transactions data
transactions_df = pd.read_csv("transactions.csv")

# Data Preprocessing by stripping whitespace from product names
transactions_df['transactions'] = transactions_df['transactions'].apply(
   lambda x: ', '.join([item.strip() for item in x.split(',')]))

# --------------- EXPLORATORY DATA ANALYSIS ------------------
# Split transactions and get a flat list of all items
all_items = [item for sublist in transactions_df['transactions'].str.split(',') for item in sublist]
item_counts = Counter(all_items)


# Average Number of Items per Transaction
transactions_df['num_items'] = transactions_df['transactions'].apply(lambda x: len(x.split(',')))
avg_items = transactions_df['num_items'].mean()


bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
transactions_df['age_group'] = pd.cut(transactions_df['customer_age'], bins=bins, labels=labels, right=False)


age_group_counts = transactions_df['age_group'].value_counts().sort_index()


# Set style for the plots
sns.set_style("whitegrid")


# Distribution of Customer Ages
plt.figure(figsize=(12, 6))
sns.histplot(transactions_df['customer_age'], bins=30, kde=True)
plt.title('Distribution of Customer Ages', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()


# Distribution of Number of Items Purchased
plt.figure(figsize=(12, 6))
sns.histplot(transactions_df['num_items'], bins=20, kde=True)
plt.title('Distribution of Number of Items Purchased', fontsize=20)
plt.xlabel('Number of Items', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()


# Boxplot of Number of Items Purchased by Age Group
plt.figure(figsize=(12, 6))
sns.boxplot(x='age_group', y='num_items', data=transactions_df, palette='pastel')
plt.title('Boxplot of Number of Items Purchased by Age Group', fontsize=20)
plt.xlabel('Age Group', fontsize=15)
plt.ylabel('Number of Items Purchased', fontsize=15)
plt.show()


# Number of Transactions by Age Group
plt.figure(figsize=(12, 6))
sns.countplot(x='age_group', data=transactions_df, palette='coolwarm')
plt.title('Number of Transactions by Age Group', fontsize=20)
plt.xlabel('Age Group', fontsize=15)
plt.ylabel('Number of Transactions', fontsize=15)
plt.show()


# Extracting top selling items and least selling items
top_selling_items = item_counts.most_common(10)
least_selling_items = item_counts.most_common()[-10:]


# Visualization for top and least selling items
plt.figure(figsize=(12, 12))
# Top selling items
labels_top, values_top = zip(*top_selling_items)
sns.barplot(x = list(labels_top), y = list(values_top), palette='viridis')
plt.title('Top 10 Most Popular Items', fontsize=20)
plt.xlabel('Products', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()


# Creating a function to get the count of items for a given age group
def get_item_counts_for_age(age_filter):
   if isinstance(age_filter, (list, tuple)):  # If age_filter is an age group
       filtered_transactions = transactions_df[transactions_df['customer_age'].between(age_filter[0], age_filter[1])]
   else:  # If age_filter is a specific age
       filtered_transactions = transactions_df[transactions_df['customer_age'] == age_filter]


   items_for_age = [item for sublist in filtered_transactions['transactions'].str.split(',') for item in sublist]
   return Counter(items_for_age)


# For the top item
top_item = labels_top[0]
top_item_counts_by_age = {age: get_item_counts_for_age(age)[top_item] for age in range(20, 101)}


# Top item counts by age
plt.figure(figsize=(12, 6))
sns.lineplot(x=list(top_item_counts_by_age.keys()), y=list(top_item_counts_by_age.values()), color='blue')
plt.title(f'Purchases of Most Popular Item ("{top_item}") by Age', fontsize=20)
plt.xlabel('Age', fontsize=15)
plt.ylabel(f'Number of Purchases of {top_item}', fontsize=15)
plt.tight_layout()
plt.show()


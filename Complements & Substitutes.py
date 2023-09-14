import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


# Read the transactions data
transactions_df = pd.read_csv("transactions.csv")


# Prepare the sales_data and product_names structures
product_names_dict = {product: idx for idx, product in enumerate(transactions_df['transactions'].unique())}
sales_data_list = [{"products": [{"product_id": product} for product in row.split(", ")]} for row in
                  transactions_df['transactions']]


# Top 50 products for consideration
top_50_products = transactions_df['transactions'].value_counts().head(50).index
transactions_df = transactions_df[transactions_df['transactions'].isin(top_50_products)]


# Function to create product purchase network matrix
def create_product_purchase_network_matrix(sales_data, product_names):
   product_indices = {product_id: idx for idx, product_id in enumerate(product_names.keys())}
   num_transactions = len(sales_data)
   num_products = len(product_names)
   biadjacency_matrix = np.zeros((num_transactions, num_products), dtype=int)
   for transaction_idx, transaction in enumerate(sales_data):
       for product in transaction["products"]:
           product_id = product["product_id"]
           product_idx = product_indices[product_id]
           biadjacency_matrix[transaction_idx, product_idx] = 1
   return biadjacency_matrix


# Optimized complementarity measure calculation
def complementarity_measure(adj_matrix):
   common_neighbors = np.dot(adj_matrix.T, adj_matrix)
   product_degrees = np.sum(adj_matrix, axis=0).reshape(-1, 1)
   simo = common_neighbors / (product_degrees @ product_degrees.T)
   np.fill_diagonal(simo, 0)
   return simo


# Optimized substitutability measure calculation
def substitutability_measure(complementarity_scores):
   np_w_c = np.array(complementarity_scores)
   np_w_c_squared_sum = np.sum(np_w_c ** 2, axis=1)
   np_w_c_transpose = np.transpose(np_w_c)
   sims = np_w_c.dot(np_w_c_transpose) / np.sqrt(np_w_c_squared_sum[:, np.newaxis] * np_w_c_squared_sum[np.newaxis, :])
   return sims


# Compute the matrices and scores
biadjacency_matrix = create_product_purchase_network_matrix(sales_data_list, product_names_dict)
complementarity_scores = complementarity_measure(biadjacency_matrix)
substitutability_scores = substitutability_measure(complementarity_scores)


# Identify top complementary and substitute products
# For this, we'll first extract the product pairs with the highest complementarity and substitutability scores
num_top_products = 10


# Get the top complementary products
complementary_products_indices = np.dstack(np.unravel_index(np.argsort(complementarity_scores.ravel()), complementarity_scores.shape))[0][-num_top_products:]
complementary_scores = [complementarity_scores[i, j] for i, j in complementary_products_indices]
complementary_products = [(list(product_names_dict.keys())[i], list(product_names_dict.keys())[j]) for i, j in complementary_products_indices]


# Get the top substitute products
substitute_products_indices = np.dstack(np.unravel_index(np.argsort(substitutability_scores.ravel()), substitutability_scores.shape))[0][-num_top_products:]
substitute_scores = [substitutability_scores[i, j] for i, j in substitute_products_indices]
substitute_products = [(list(product_names_dict.keys())[i], list(product_names_dict.keys())[j]) for i, j in substitute_products_indices]


print(complementary_products)
print(complementary_scores)
print(substitute_products)
print(substitute_scores)
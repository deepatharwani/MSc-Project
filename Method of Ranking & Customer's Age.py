import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')


# Read the transactions data
transactions_df = pd.read_csv("transactions.csv")


# Data Preprocessing by stripping whitespace from product names
transactions_df['transactions'] = transactions_df['transactions'].apply(
   lambda x: ', '.join([item.strip() for item in x.split(',')]))


# --------------- RANKING OF PRODUCTS ------------------


print("\n RESULTS FOR THE RANKING BASED METHOD:\n")


# Constructing the bipartite graph
B_clean = nx.Graph()


# Add nodes to the bipartite graph
transaction_nodes_clean = transactions_df['trans_id'].unique().tolist()
B_clean.add_nodes_from(transaction_nodes_clean, bipartite=0)  
product_nodes_clean = set() 


for _, row in transactions_df.iterrows():
   products = row['transactions'].split(', ')
   product_nodes_clean.update(products)


   for product in products:
       B_clean.add_edge(row['trans_id'], product) # Adding edges to the network


B_clean.add_nodes_from(product_nodes_clean, bipartite=1)


# Create the weighted graph using cleaned data
product_network_clean = bipartite.weighted_projected_graph(B_clean, product_nodes_clean)


# Print Nodes and Edges of the projected network
num_nodes_product_clean = product_network_clean.number_of_nodes()
num_edges_product_clean = product_network_clean.number_of_edges()


print("Number of Nodes: ",num_nodes_product_clean,"\nNumber of Edges: ", num_edges_product_clean,"\n")


# Compute the degrees of nodes in the product_network_clean
degrees = dict(product_network_clean.degree())
degree_values = sorted(set(degrees.values()))
degree_hist = [list(degrees.values()).count(i) / float(nx.number_of_nodes(product_network_clean)) for i in degree_values]


# Extracting edges with their weights (co-purchase frequencies) from the network
edges_with_weights_clean = [(u, v, d['weight']) for u, v, d in product_network_clean.edges(data=True)]


# Sorting edges based on weights in descending order to get top co-purchased product pairs
sorted_edges_by_weight_clean = sorted(edges_with_weights_clean, key=lambda x: x[2], reverse=True)


# Identifying potential substitutes using Jaccard similarities
def compute_jaccard_similarity(set1, set2):
   """Compute Jaccard Similarity between two sets."""
   intersection_len = len(set1.intersection(set2))
   union_len = len(set1.union(set2))
   if union_len == 0:
       return 0
   return intersection_len / union_len


# Identifying customer base for each product
product_customer_map = {}
for product in product_nodes_clean:
   product_customer_map[product] = set(B_clean.neighbors(product))


product_pairs_info = []
for product1 in product_nodes_clean:
   for product2 in product_nodes_clean:
       if product1 != product2:
           jaccard_sim = compute_jaccard_similarity(product_customer_map[product1], product_customer_map[product2])
           co_purchase_frequency = product_network_clean[product1][product2]['weight'] if product_network_clean.has_edge(product1, product2) else 0
           product_pairs_info.append((product1, product2, jaccard_sim, co_purchase_frequency))


# Sort product pairs by Jaccard similarity (descending)
product_pairs_info_sorted = sorted(product_pairs_info, key=lambda x: x[2], reverse=True)


# Filter potential substitutes: high Jaccard similarity but low co-purchase frequency
potential_substitutes = [pair for pair in product_pairs_info_sorted if pair[3] < 3 and pair[2] > 0.005]


# Top 10 potential substitutes 
top_potential_substitutes = potential_substitutes[:10]


# Sorting product pairs alphabetically to prevent duplicates
def sort_product_pair(product_pair):
   products = product_pair.split(" & ")
   return " & ".join(sorted(products))


# Creating a table for complementary and substitute products
table_data = []
table_data_sub = []


# Add data from sorted_edges_by_weight_clean i.e. complementary products
for item in sorted_edges_by_weight_clean:
   table_data.append((" & ".join(item[:2]), item[2]))


# Add data from top_potential_substitutes
for item in top_potential_substitutes:
   sorted_product_pair = sort_product_pair(" & ".join(item[:2]))
   table_data_sub.append((sorted_product_pair, item[3]))


# Convert to DataFrame for tabular display
df_display = pd.DataFrame(table_data, columns=["Complementary Products", "Count"])
df_display_sub = pd.DataFrame(table_data_sub, columns=["Substitute Products", "Count"]).drop_duplicates().reset_index(drop=True)


print(df_display.head(10),'\n')
print(df_display_sub.sort_values('Count', ascending=1).head(10))


# Extracting the top 10 complementary and substitute products
top_complementary_products = df_display.head(10)
top_substitute_products = df_display_sub.sort_values('Count', ascending=1).head(10)

# Get the list of top products for the below method
top_complementary_product_list = [item.split(" & ") for item in top_complementary_products["Complementary Products"].tolist()]
top_substitute_product_list = [item.split(" & ") for item in top_substitute_products["Substitute Products"].tolist()]


# Getting unique product names
unique_top_products = list(set([item for sublist in top_complementary_product_list + top_substitute_product_list for item in sublist]))


# Creating a dictionary to map product names to indices
product_to_index = {product: index for index, product in enumerate(unique_top_products)}


# Initialize a matrix to hold co-purchase frequencies
num_top_products = len(unique_top_products)
co_purchase_matrix = np.zeros((num_top_products, num_top_products), dtype=int)


# Fill the matrix with co-purchase frequencies
for product1 in unique_top_products:
   for product2 in unique_top_products:
       if product1 != product2:
           if product_network_clean.has_edge(product1, product2):
               co_purchase_matrix[product_to_index[product1], product_to_index[product2]] = product_network_clean[product1][product2]['weight']


# Convert the matrix to a DataFrame for easier handling
co_purchase_df = pd.DataFrame(co_purchase_matrix, index=unique_top_products, columns=unique_top_products)

# --------------- CUSTOMER'S AGE TO PRODUCT RELATIONSHIP ----------------

print("\n RESULTS FOR CUSTOMER'S AGE TO PRODUCT RELATIONSHIP:\n")

bins = [15, 25, 35, 45, 55, 65]
labels = ['16-25', '26-35', '36-45', '46-55', '56-65']


# Grouping the customer ages into bins
transactions_df['age_group'] = pd.cut(transactions_df['customer_age'], bins=bins, labels=labels, right=True)


# Purchasing patterns by age group
age_group_purchases = transactions_df.groupby('age_group')['transactions'].apply(lambda x: ','.join(x)).str.split(',', expand=True).stack()
age_group_purchases_counts = age_group_purchases.reset_index(level=1, drop=True).reset_index(name='Product').groupby(['age_group', 'Product']).size().reset_index(name='Count')


# Identifying the top purchased products for each age group
top_purchased_products_by_age = age_group_purchases_counts.groupby('age_group').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)


# Splitting the dataframe into individual tables for each age group
age_groups = top_purchased_products_by_age['age_group'].unique()


# Dictionary for top 5 complement products
age_group_dfs = {}
for age in age_groups:
   age_group_dfs[age] = top_purchased_products_by_age[top_purchased_products_by_age['age_group'] == age]


# Print the age group results
print("\n")
for age, products_df in age_group_dfs.items():
   print(f"Age Group: {age}\n")
   print(products_df)
   print("\n")


# Visualizing the purchasing patterns by age group
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))


# Defining the parameters for visualization
palette = sns.color_palette("viridis", n_colors=10)


# Plot the top 5 purchased products for each age group
for ax, age_group in zip(axes.flatten(), age_groups):
   df_age = age_group_dfs[age_group]
   sns.barplot(data=df_age, x="Count", y="Product", ax=ax, palette=palette)
   ax.set_title(f"Age {age_group}", fontsize=10)
   ax.set_xlabel("No. of Products")
   ax.set_ylabel("Products")
   ax.tick_params(labelsize=10)


fig.delaxes(axes[2, 1])  # Remove the unused subplot
fig.suptitle("Top 5 Products Purchased by Age Group", fontsize=10, y=1.03)
plt.tight_layout()
plt.show()


# Constructing the bipartite graph between age groups and complementary products
B_age_product = nx.Graph()


# Add nodes for age groups
B_age_product.add_nodes_from(age_groups, bipartite=0)


# Extracting top complementary products for each age group
top_complementary_products = df_display["Complementary Products"].head(10).tolist()


# Add nodes for top complementary products
B_age_product.add_nodes_from(top_complementary_products, bipartite=1)


# Determine if a complementary product pair has been co-purchased by a specific age group
def co_purchased_by_age_group(product_pair, age_group_df):
   product1, product2 = product_pair.split(" & ")
   co_purchase_count = age_group_df[age_group_df['transactions'].str.contains(product1, regex=False) & age_group_df['transactions'].str.contains(product2, regex=False)].shape[0]
   return co_purchase_count > 0


# Add edges between age groups and complementary products based on co-purchase data
for age_group in age_groups:
   age_group_df = transactions_df[transactions_df['age_group'] == age_group]
   for product_pair in top_complementary_products:
       if co_purchased_by_age_group(product_pair, age_group_df):
           B_age_product.add_edge(age_group, product_pair)


# Drawing the bipartite graph
pos = nx.bipartite_layout(B_age_product, age_groups)
plt.figure(figsize=(12, 8))
nx.draw(B_age_product, pos, with_labels=True, node_color=['skyblue']*len(age_groups)+['lightgreen']*len(top_complementary_products), node_size=2000, font_size=10, edge_color="gray")
plt.title("Bipartite Network between Age Groups and Complementary Products")
plt.show()

# Extracting top substitute products
top_substitute_products = [pair[0] + " & " + pair[1] for pair in top_potential_substitutes]


B_age_product_sub = nx.Graph()
B_age_product_sub.add_nodes_from(age_groups, bipartite=0)
B_age_product_sub.add_nodes_from(top_substitute_products, bipartite=1)


for age_group in age_groups:
   age_group_df = transactions_df[transactions_df['age_group'] == age_group]
   for product_pair in top_substitute_products:
       if co_purchased_by_age_group(product_pair, age_group_df):
           B_age_product_sub.add_edge(age_group, product_pair)


# Drawing the substitute products bipartite graph
pos = nx.bipartite_layout(B_age_product_sub, age_groups)
plt.figure(figsize=(12, 8))
nx.draw(B_age_product_sub, pos, with_labels=True, node_color=['skyblue']*len(age_groups)+['lightcoral']*len(top_substitute_products), node_size=2000, font_size=10, edge_color="gray")
plt.title("Bipartite Network between Age Groups and Substitute Products")
plt.show()


# ------------------- BASIC SUPERVISED LEARNING METHOD FOR PREDICTING AGE GROUPS -----------------


print("\n RESULTS FOR RANDOM FOREST CLASSIFIER:\n")


# Step 1: Data Preparation
df = pd.read_csv('transactions.csv')


# Drop rows with missing values
df = df.dropna()


# Creating age groups
bins = [15, 25, 35, 45, 55, 65]
labels = ['16-25', '26-35', '36-45', '46-55', '56-65']
df['age_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels, right=True)


# Drop rows with missing age_group values
df = df.dropna(subset=['age_group'])


# Creating product pairs as features
df['transactions'] = df['transactions'].apply(lambda x: ', '.join([item.strip() for item in x.split(',')]))
df['product_pairs'] = df['transactions'].apply(lambda x: [f'{item[0]} & {item[1]}' for item in combinations(x.split(', '), 2)])


# Step 2: Feature Engineering
rows = []
for idx, row in df.iterrows():
   for pair in row['product_pairs']:
       rows.append([row['trans_id'], row['customer_age'], row['age_group'], pair])


df_pairs = pd.DataFrame(rows, columns=['trans_id', 'customer_age', 'age_group', 'product_pair'])


# Step 3: Label Generation
# Using age group as the label
X = df_pairs['product_pair']
y = df_pairs['age_group']


# Step 4: Model Selection and Training
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Applying one-hot encoding to the product pairs
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Using a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# Step 5: Evaluation
# Predicting on the test set and evaluating the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


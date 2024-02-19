# Advanced-A-1
#Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##Upload the csv data
data = pd.read_csv('/content/Youth_Tobacco_Survey__YTS__Data.csv')

# Display the first few rows of the dataset 
print("First few rows of the dataset:") 
print(data.head()) 

# Summary statistics 
print("\nSummary statistics:") 
print(data.describe()) 

###MISSING VALUES 
# Check for missing values 
print("\nMissing values:") 
print(data.isnull().sum())

###REMOVAL OF MISSING VALUES ON NON-NUMERICAL DATA
non_numeric_columns = data.select_dtypes(exclude=['number']).columns

# Remove missing values from non-numeric columns
cleaned_data = data.dropna(subset=non_numeric_columns)
numerical_columns = ['YEAR', 'Data_Value', 'Data_Value_Std_Err', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']

# Check if there are any numerical columns
if cleaned_data.empty:
print("No data remaining after removing missing values from non-numeric columns.")
elif numerical_columns:

# Instantiate SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Impute missing values in numerical columns
cleaned_data[numerical_columns] = imputer.fit_transform(cleaned_data[numerical_columns])
else:
print("No numerical columns found for imputation.")
OUTPUT: No data remaining after removing missing values from non-numeric columns.
numerical_columns = ['YEAR', 'Data_Value', 'Data_Value_Std_Err', 'Low_Confidence_Limit', 'High_Confidence_Limit', 'Sample_Size']

# Check if there are any numerical columns
if cleaned_data.empty:
print("No data remaining after removing missing values from non-numeric columns.")
elif numerical_columns:
if cleaned_data[numerical_columns].shape[0] > 0:

# Instantiate StandardScaler
scaler = StandardScaler()

# Scale the numerical columns
cleaned_data[numerical_columns] = scaler.fit_transform(cleaned_data[numerical_columns])
else:
print("No valid samples remaining in numerical columns after removing missing values.")
else:
print("No numerical columns found for scaling.")
OUTPUT: No data remaining after removing missing values from non-numeric columns.
print("\nMissing values after cleaning:")
print(cleaned_data.isnull().sum())

# Save the cleaned dataset
data.to_csv('clean_data.csv', index=False)
print("\nFirst few rows of the cleaned dataset:")
print(data.head()) 

###CLUSTERING
cleaned_data = pd.read_csv('/content/clean_data.csv')
numerical_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
numerical_data = cleaned_data[numerical_columns]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Instantiate SimpleImputer to impute missing values with mean
imputer = SimpleImputer(strategy='mean')

# Impute missing values
scaled_data_imputed = imputer.fit_transform(scaled_data)

#Elbow Method
inertia = []
for n_clusters in range(1, 11):
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data_imputed)
inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show() 

optimal_n_clusters = 3

# Adjust based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data_imputed)

# Add cluster labels to the original dataframe
data['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_n_clusters):
plt.scatter(scaled_data_imputed[data['Cluster'] == cluster][:, 0],
scaled_data_imputed[data['Cluster'] == cluster][:, 1],
label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
marker='x', color='k', label='Centroids', s=100)
plt.title('Cluster Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

# Save clustered data
data.to_csv('clustered_data.csv', index=False)

##Evaluate cluster quality
silhouette_avg = silhouette_score(scaled_data_imputed, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")

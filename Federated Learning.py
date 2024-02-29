#FEDERATED MODELLING 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split

#USER DEFINED FUNCTION
# Function to plot 3D scatter plot
def plot_3d_scatter(X, labels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', edgecolor='k', s=50)
    
    st.write("3D Scatter Plot of Cancer Gene Expression RNA-Seq with PCA Components")
    # Adding labels
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    
    # Adding colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    
    plt.title("3D Scatter Plot of Cancer Gene Expression RNA-Seq with PCA Components")
    st.pyplot(plt)

#READING THE INPUT DATA
X = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\X_pca.csv",header=None)

Y= pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\y_balanced.csv")

data = pd.concat([X, Y], axis=1)

# PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Federated Modelling",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for wider view
)

# HEADER SECTION
st.title("FEDERATED LEARNING ALGORITHM WITH K- MEANS CLUSTERING MODEL")
#PRINTING THE DIMENSIONS OF THE DATASET
st.markdown("### Dataset Dimensions")
st.write("Rows:", X.shape[0])
st.write("Columns:", X.shape[1])
st.write("The size of the dataset is : 801 x 16384")

st.write("Since out K = 5.Let the number of sub models constructed be 5")

DF1 = pd.DataFrame()
DF2 = pd.DataFrame()
DF3 = pd.DataFrame()
DF4 = pd.DataFrame()
DF5 = pd.DataFrame()

# Number of subsets
num_subsets = 5

subsets = []

# Shuffle the data to ensure randomness
data = data.sample(frac=1, random_state=42)

# Calculate the size of each subset
subset_size = len(data) // num_subsets

# Split the data into 5 equal subsets
for i in range(num_subsets):
    start_idx = i * subset_size
    end_idx = (i + 1) * subset_size
    subset = data.iloc[start_idx:end_idx]
    subsets.append(subset)

# Ensure that each subset is balanced
for i, subset in enumerate(subsets):
    class_distribution = subset['Class'].value_counts()
    print(f"Subset {i + 1} class distribution:\n{class_distribution}")
    # Ensure the class distribution is balanced as well

# Assign subdatasets to your DataFrames
DF1 = subsets[0]
DF2 = subsets[1]
DF3 = subsets[2]
DF4 = subsets[3]
DF5 = subsets[4]

DF1 = DF1.drop(columns=['Class'])
DF2 = DF2.drop(columns=['Class'])
DF3 = DF3.drop(columns=['Class'])
DF4 = DF4.drop(columns=['Class'])
DF5 = DF5.drop(columns=['Class'])

#STORING THE MEAN PARAMETERS IN THE LIST
Parameters=[]

DFs=[DF1,DF2,DF3,DF4,DF5]

for j,df in enumerate(DFs):
    st.markdown("###  K-MEANS CLUSTERING ALGORITHM FOR K =5 SUBMODEL "+str(j))
    # Fit the K-Means model for K=5
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df.iloc[:, :2])
    
    # Cluster assignments for each data point
    cluster_labels = kmeans.labels_
    # Applying PCA with 3 principal components
    pca_3d = PCA(n_components=3, random_state=50)
    X_pca_3d = pca_3d.fit_transform(df)
    # Plotting 3D scatter plot for the clusters
    plot_3d_scatter(X_pca_3d,cluster_labels)
    
    st.write(" 2D Scatter Plot of Cancer Gene Expression RNA-Seq with PCA Components")
    plt.figure(figsize=(10, 8))
    # plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='binary', edgecolor='k', s=50)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], edgecolor='k', s=50)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster Label")
    st.pyplot(plt)
    st.write(" MODEL PARAMETERS")
    # PRINTING THE PARAMETERS
    st.write("Cluster Labels : ", cluster_labels)
    # Cluster centers
    cluster_centers = kmeans.cluster_centers_
    st.write("Cluster Centers : ", cluster_centers)
    # Inertia (within-cluster sum of squares)
    inertia = kmeans.inertia_
    st.write("Inertia within cluster : ", inertia)
    # Number of iterations required for convergence
    n_iter = kmeans.n_iter_
    st.write("Number of Iterations required for convergence : ", n_iter)
    Parameters.append(cluster_centers)


#PRINTING ALL THE MEAN PARAMETERS TOGETHER
st.markdown("PARAMETERS OF ALL THE SUBMODELS")
st.write(Parameters)

Final_parameters=[]

# Aggregating all the parameters using average
Final_parameters2 = [sum(val) / len(val) for val in zip(*Parameters)]
print(Final_parameters2)

Final=[]

for i in range(5):
    Coordinates=[]
    for j in range(2):
        val=0
        for k in range(5):
            val+=Parameters[i][k][j]
        Coordinates.append(val/5)
    Final.append(Coordinates)  

print(Final)
st.write("Aggregated Cluster Centers : ",Final)
data=data.drop(columns=['Class'])

# Fit the K-Means model for K=5 on the entire dataset
kmeans = KMeans(n_clusters=5)
kmeans.fit(data.iloc[:, :2])

cluster_centers = kmeans.cluster_centers_
print(cluster_centers)
st.write("Global Cluster Centers : ",cluster_centers)
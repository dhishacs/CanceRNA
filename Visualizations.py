#VISUALISATIONS
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
from scipy import stats

#READING THE INPUT DATA
X = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\data.csv")
# Drop the first column
X = X.drop(X.columns[0], axis=1)

#READING THE CLASSIFICATION LABELS
Y = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\labels.csv")
# Drop the first column
Y = Y.drop(Y.columns[0], axis=1)

# PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Visualizations",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for wider view
)

# HEADER SECTION
st.title("VISUALIZATIONS")

# Plotting a simple histogram to show the distribution of the numerical values of a gene
st.markdown("### Distribution of a Gene")
selected_column = st.selectbox("Select a Gene", X.select_dtypes(include="number").columns)
plt.figure(figsize=(8, 6))
plt.hist(X[selected_column], bins=20, color="skyblue", edgecolor="black")
plt.xlabel(selected_column)
plt.ylabel("Frequency")
st.pyplot(plt)

# Displaying the pie plot for the distribution of the tumors
st.markdown("### Distribution of Tumors")
values_distribution = Y.iloc[:, 0].value_counts()
num_categories = len(values_distribution)
colors = plt.cm.tab10.colors[:num_categories]  # Using the 'tab10' colormap for distinct colors
plt.figure(figsize=(8, 6))
plt.pie(values_distribution, labels=values_distribution.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Distribution of Tumors")
st.pyplot(plt)

#Displaying boxplots for the selected column
selected_column_boxplot = st.selectbox("Select a Column for Box Plot", X.columns)
st.markdown(f"### Box Plot with Outliers for {selected_column_boxplot}")
plt.figure(figsize=(8, 6))
plt.boxplot(X[selected_column_boxplot])
plt.title(f"Box Plot for {selected_column_boxplot}")
plt.xlabel("Value")
plt.ylabel("Frequency")
st.pyplot(plt)

#Outlier removal
z_scores = np.abs(stats.zscore(X))
threshold = 3
outlier_indices = np.where(z_scores > threshold)

# Remove outliers
# new DataFrame with outliers removed
X = X[(z_scores < threshold).all(axis=1)]

# Displaying a count plot for the first column of the dataset Y
st.markdown("### Count Plot for Tumors")
plt.figure(figsize=(8, 6))
sns.countplot(x=Y.iloc[:, 0])
plt.xlabel("Categories")
plt.ylabel("Count")
st.pyplot(plt)

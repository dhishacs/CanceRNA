import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# USER DEFINED FUNCTIONS
# Function to display a portion of the DataFrame
def display_dataframe(df, num_rows=10):
    st.dataframe(df.head(num_rows))

# Function to display a subset of columns
def display_dataframe_columns(df, num_columns=10):
    st.dataframe(df.iloc[:, :num_columns])

# READING THE DATAFRAME
X = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\data.csv")

# PAGE CONFIGURATIONS
st.set_page_config(
    page_title="ML CA 3 - Hackathon",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for wider view
)

# HEADER SECTION
st.title("Welcome to the ML CA 3 Hackathon")
st.subheader("Main Page")
st.sidebar.success("Select a page above.")

# ABOUT THE DATASET
st.markdown("### About the Dataset")
st.write("This collection of data is part of the RNA-Seq (HiSeq) Pan-Cancer Atlas dataset, it is a random extraction of gene expressions of patients having different types of tumors")
st.write("The different types of tumors are : ")
st.write("                                   BRCA — Breast invasive carcinoma")
st.write("                                   KIRC — Kidney renal clear cell carcinoma")
st.write("                                   COAD — Colon adenocarcinoma")
st.write("                                   LUAD — Lung adenocarcinoma")
st.write("                                   PRAD — Prostate adenocarcinoma")

# Display a portion of the DataFrame
st.markdown("### Preview of the Dataset")
display_dataframe(X)

# Display a subset of columns
st.markdown("### Subset of Columns")
display_dataframe_columns(X)

# Add an expander for displaying the entire DataFrame
with st.expander("View Entire Dataset"):
    st.dataframe(X)


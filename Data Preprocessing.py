#DATA PREPROCESSING
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# READING THE DATAFRAME
X = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\data.csv")

# PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Data Preprocessing",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for wider view
)

# HEADER SECTION
st.title("DATA PREPROCESSING")

#PRINTING THE DIMENSIONS OF THE DATASET
st.markdown("### Dataset Dimensions")
st.write("Rows:", X.shape[0])
st.write("Columns:", X.shape[1])
st.write("The size of the dataset is : 801 x 16384")

# DISPLAYING COLUMN NAMES
st.markdown("### Column Names")
column_names = X.columns.tolist()
st.write(column_names)

# DISPLAYING INFO ABOUT THE DATASET
st.markdown("### Dataset Information")
# Capture the output of X.info() and display it
with StringIO() as buffer:
    X.info(buf=buffer)
    info_text = buffer.getvalue()

st.text(info_text)
st.write("We need to drop the Untitled Column.")
# Drop the first column
X = X.drop(X.columns[0], axis=1)

# DISPLAYING SUMMARY STATISTICS
st.markdown("### Summary Statistics")
# Capture the output of X.describe() and display it
with StringIO() as buffer_describe:
    X.describe().to_string(buf=buffer_describe)
    describe_text = buffer_describe.getvalue()
st.text(describe_text)
st.write("From the summary we can see that the values ranges from 0 to 10-15 for all the genes.")

# DISPLAYING DATATYPES
st.markdown("### Data Types")
# Capture the output of DF.dtypes and display it
with StringIO() as buffer_dtypes:
    X.dtypes.to_string(buf=buffer_dtypes)
    dtypes_text = buffer_dtypes.getvalue()
st.text(dtypes_text)
st.write("We can see that all the columns are of type float or int.")

# CHECKING FOR DUPLICATES
st.markdown("### Checking for Duplicates")
duplicates_info = X.duplicated().value_counts()
st.write("Number of Duplicates:")
st.write(duplicates_info)
st.write("We can see that there are no duplicates in the dataset.")

# CHECKING FOR NULL VALUES
st.markdown("### Checking for Null Values")
null_values = X.isnull().sum()
st.write("Number of Null Values in Each Column:")
st.write(null_values)
st.write("We can see the count of null values in each column.Notice that there are no null values in the dataset.")

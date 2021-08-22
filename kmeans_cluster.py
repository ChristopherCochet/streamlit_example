# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Helper functions
def run_kmeans(df, n_clusters=2):
    '''  Create scatterplot from the Kmean clustering '''
    kmeans = KMeans(n_clusters, random_state= 42).fit(df[["Age", "Income"]])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax = sns.scatterplot(
        ax=ax,
        x=df.Age,
        y=df.Income,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=n_clusters),
        legend=None,
    )

    return fig

@st.cache
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv"
    )
    return df

# Laod the dataframe
df = load_data()

# Sidebar
sidebar = st.sidebar
df_display = sidebar.checkbox("Display Raw Data", value=True)

n_clusters = sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=10,
)

# Title
st.title("Interactive K-Means Clustering")
# Description
st.write("Here is the raw verison of the dataset used for the clustering:")
# Display the dataframe in tabular format
if df_display:
    st.write(df)

# Show cluster scatter plot
st.write(run_kmeans(df, n_clusters=n_clusters))
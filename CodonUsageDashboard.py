
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from scipy.stats import entropy

# Load all data files
@st.cache_data
def load_data():
    human_229e = pd.read_excel("codon_counts_per_sequence_Human_229E_CoronaVirus (1).xlsx")
    human_nl63 = pd.read_excel("codon_counts_per_sequence_Human_NL63_CoronaVirus (2).xlsx")
    cattle = pd.read_excel("codon_counts_per_sequence_cattleCoronaVirus (1).xlsx")
    dog = pd.read_excel("codon_counts_per_sequence_DogsCoronaVirus (1).xlsx")

    # Add host labels
    human_229e["Host"] = "Human_229E"
    human_nl63["Host"] = "Human_NL63"
    cattle["Host"] = "Cattle"
    dog["Host"] = "Dog"

    # Combine
    df = pd.concat([human_229e, human_nl63, cattle, dog], ignore_index=True)
    df.drop(columns=["Sequence ID"], inplace=True, errors="ignore")
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Overview", "EDA", "Correlation", "ML Results", "Entropy & PCA", "Conclusion"])

if selection == "Overview":
    st.title("Codon Usage Analysis of Coronaviruses")
    st.markdown("""
    This dashboard explores codon usage in coronavirus genomes from different hosts (Human, Dog, Cattle).

    **Goals:**
    - Analyze codon distribution
    - Compare entropy across viruses
    - Visualize host-specific patterns
    - Build ML models to classify and cluster viruses

    **Dataset:** Provided Excel files containing codon counts per genome.
    """)

elif selection == "EDA":
    st.header("Exploratory Data Analysis")
    host = st.selectbox("Select Host Group", df["Host"].unique())
    codon = st.selectbox("Select Codon to Visualize", df.columns[:-1])

    filtered = df[df["Host"] == host]
    fig, ax = plt.subplots()
    sns.histplot(filtered[codon], kde=True, ax=ax)
    ax.set_title(f"Distribution of {codon} in {host}")
    st.pyplot(fig)

elif selection == "Correlation":
    st.header("Codon Correlation Heatmap")
    selected_host = st.selectbox("Choose a Host for Heatmap", ["All Combined"] + list(df["Host"].unique()))
    if selected_host == "All Combined":
        temp = df.copy()
    else:
        temp = df[df["Host"] == selected_host]
    features = temp.drop(columns=["Host"])
    norm = features.div(features.sum(axis=1), axis=0)
    corr = norm.corr()

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif selection == "ML Results":
    st.header("Model Performance Summary")
    st.markdown("""
    - **Regression**: Entropy prediction using codons.
    - **Classification**: Host classification using Random Forest.
    - **Clustering**: KMeans applied to normalized codons.

    *All training and evaluation is handled in Colab. This section presents final metrics.*

    **Regression MSE:** 0.0036  
    **Classification Accuracy:** 93%  
    **Clustering Silhouette Score:** 0.62
    """)

elif selection == "Entropy & PCA":
    st.header("Entropy vs Host")
    features = df.drop(columns=["Host"])
    norm = features.div(features.sum(axis=1), axis=0)
    df["Entropy"] = norm.apply(lambda row: entropy(row + 1e-8), axis=1)

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Host", y="Entropy", ax=ax)
    st.pyplot(fig)

    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(norm)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["Host"] = df["Host"].values
    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Host", title="PCA of Codon Usage")
    st.plotly_chart(fig)

elif selection == "Conclusion":
    st.title("Insights and Conclusions")
    st.markdown("""
    ✅ Codon usage patterns are highly host-specific.

    ✅ Entropy varies significantly between human and animal viruses.

    ✅ PCA and clustering show that codon usage forms distinct biological clusters.

    ✅ Classification models can successfully predict host species using only codon frequencies.

    ✅ This dashboard offers interactive views for exploring viral codon behavior and ML findings.
    """)

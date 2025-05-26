# Enhanced Streamlit dashboard with biological context and improved interactivity

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy
import plotly.express as px

# Load Data
def load_data():
    human_229e = pd.read_excel("codon_counts_per_sequence_Human_229E_CoronaVirus (1).xlsx")
    human_nl63 = pd.read_excel("codon_counts_per_sequence_Human_NL63_CoronaVirus (2).xlsx")
    cattle = pd.read_excel("codon_counts_per_sequence_cattleCoronaVirus (1).xlsx")
    dog = pd.read_excel("codon_counts_per_sequence_DogsCoronaVirus (1).xlsx")

    human_229e["Host"] = "Human_229E"
    human_nl63["Host"] = "Human_NL63"
    cattle["Host"] = "Cattle"
    dog["Host"] = "Dog"

    df = pd.concat([human_229e, human_nl63, cattle, dog], ignore_index=True)
    df.drop(columns=["Sequence ID"], inplace=True, errors="ignore")
    return df

df = load_data()

# Bio-themed page background
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] > .main {
background-image: url("https://www.genome.gov/sites/default/files/media/images/2020-01/codon-hero.jpg");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}
[data-testid="stSidebar"] > div:first-child {
background-color: #F0F2F6;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧬 Codon Usage Explorer")
page = st.sidebar.radio("Navigate to:", [
    "🏠 Overview",
    "📊 EDA",
    "📈 Correlation",
    "🧠 ML Results",
    "🌐 Entropy & PCA",
    "🧾 Biological Summary"
])

if page == "🏠 Overview":
    st.title("🧬 Coronavirus Codon Usage Dashboard")
    st.markdown("""
    Welcome to the **Codon Usage Explorer**, an interactive dashboard to explore how various coronaviruses use codons based on their host species.

    **Why Codons?**  
    Codons are 3-letter sequences that form the genetic code. Each organism (or virus) may prefer certain codons due to evolutionary adaptation to its host. This dashboard uncovers those patterns.

    **Hosts Studied:**
    - Human Coronavirus 229E
    - Human Coronavirus NL63
    - Cattle Coronavirus
    - Canine Coronavirus

    Navigate using the sidebar to view distribution, relationships, and biological patterns in codon usage.
    """)

elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    selected_host = st.selectbox("Choose a host:", df["Host"].unique())
    selected_codon = st.selectbox("Select a codon:", df.columns[:-1])

    host_data = df[df["Host"] == selected_host]
    st.subheader(f"Distribution of codon '{selected_codon}' in {selected_host}")
    fig, ax = plt.subplots()
    sns.histplot(host_data[selected_codon], kde=True, ax=ax)
    st.pyplot(fig)

elif page == "📈 Correlation":
    st.header("📈 Codon Correlation Heatmap")
    selected = st.selectbox("Host-specific correlation", ["All Combined"] + list(df["Host"].unique()))
    sub_df = df if selected == "All Combined" else df[df["Host"] == selected]
    codon_data = sub_df.drop(columns="Host")
    codon_freq = codon_data.div(codon_data.sum(axis=1), axis=0)
    corr_matrix = codon_freq.corr()

    st.subheader(f"Codon Correlation Heatmap - {selected}")
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif page == "🧠 ML Results":
    st.header("🧠 Model Summary (from notebook results)")
    st.success("Entropy predicted from codons using Linear Regression (MSE: 0.0036)")
    st.success("Random Forest Classification Accuracy: 93% (4 host classes)")
    st.info("KMeans Clustering produced 4 distinct clusters with silhouette score: 0.62")
    st.markdown("Visuals are available in the PCA section.")

elif page == "🌐 Entropy & PCA":
    st.header("🌐 Entropy and Codon Usage Diversity")
    codon_data = df.drop(columns="Host")
    codon_freq = codon_data.div(codon_data.sum(axis=1), axis=0)
    df["Entropy"] = codon_freq.apply(lambda row: entropy(row + 1e-8), axis=1)

    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Host", y="Entropy", ax=ax)
    st.subheader("Entropy across Hosts")
    st.pyplot(fig)

    st.subheader("PCA of Codon Usage")
    pca_model = PCA(n_components=2)
    pcs = pca_model.fit_transform(codon_freq)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    pca_df["Host"] = df["Host"]
    fig2 = px.scatter(pca_df, x="PC1", y="PC2", color="Host", title="PCA Plot")
    st.plotly_chart(fig2)

elif page == "🧾 Biological Summary":
    st.title("🧾 Final Insights")
    st.markdown("""
    **🧠 Biological Takeaways:**
    - Viruses adapt codon usage to match their host's tRNA availability.
    - Entropy reveals which viruses optimize or diversify codons.
    - Codon preferences reflect evolutionary pressure and host-specific tuning.

    **🧬 ML Learnings:**
    - Host classification from codons is highly accurate.
    - Entropy is predictable from codon patterns.
    - PCA shows separation between virus groups.

    This dashboard combines bioinformatics with AI to explore viral adaptation.
    """)

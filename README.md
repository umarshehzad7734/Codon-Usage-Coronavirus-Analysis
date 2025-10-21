# Codon Usage Based Analysis and Host Prediction of Coronavirus Genomes

## Project Overview
This repository contains the code and resources for a comprehensive computational study focusing on **codon usage patterns** to predict the host species of various coronaviruses. By employing a pipeline that combines bioinformatics and machine learning, this project aims to demonstrate that codon usage bias is a robust genomic marker for understanding viral evolution and host adaptation.

The study utilized an Excel dataset containing codon usage data from four hosts: **human (229E and NL63), dog, and cattle coronaviruses**.

## Key Findings & Results

The computational analysis yielded high-performance results across the applied models:

* **Host Prediction:** A **Random Forest** classifier was used for supervised classification, achieving an impressive **93% accuracy** in host prediction, confirming codon usage bias as a powerful indicator.
* **Codon Entropy Prediction:** A **Linear Regression** model was applied to predict codon entropy, resulting in a strong fit with **value of approximately 0.89**.
* **Viral Grouping:** **K-Means clustering** was used for unsupervised analysis, which revealed distinct viral groupings with a **Silhouette score of approximately 0.62**.

## Methodology and Implementation

The project followed a robust computational pipeline, including:
1.  Data Preprocessing: Cleaning and normalizing the raw codon usage data.
2.  Feature Engineering: Calculating genomic markers such as Codon Adaptation Index (CAI) or Relative Synonymous Codon Usage (RSCU), as well as **entropy calculations**.
3.  Model Application: Implementation of the following machine learning models:
    Classification: Random Forest (for host prediction)
    Regression:Linear Regression (for entropy prediction)
    Clustering: K-Means (for viral grouping)

## Technologies Used

| Tool/Library | Purpose |
| **Python 3.x** | Primary programming language |
| **Pandas / NumPy** | Data manipulation and numerical operations |
| **Scikit-learn** | Implementing Machine Learning models (Random Forest, K-Means, Linear Regression) |


### Installation and Setup

1.  **Clone the repository:**

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    * Provide a command to run your main script.

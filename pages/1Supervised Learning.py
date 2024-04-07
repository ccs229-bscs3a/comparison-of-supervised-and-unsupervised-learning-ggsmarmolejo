#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics
import time

# Define the Streamlit app
def app():

    st.subheader('Supervised Learning, Classification, and KNN with Diabetes Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks. 
    \n**The Diabetes Dataset:**
    The Diabetes dataset is a popular benchmark dataset in machine learning. It contains information about 442 
    diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline.
    Attribute Information of diabetes patients:
    * age age in years
    * sex
    * bmi body mass index
    * bp average blood pressure
    * s1 tc, total serum cholesterol
    * s2 ldl, low-density lipoproteins
    * s3 hdl, high-density lipoproteins
    * s4 tch, total cholesterol / HDL
    * s5 ltg, possibly log of serum triglycerides level
    * s6 glu, blood sugar level
    
    \n**KNN Classification with Diabetes:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Diabetes dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with a new data, KNN calculates the distance (often Euclidean distance) 
    between the patient's attribute and the disease progression.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest 
    data points (patients) in the training set to the new flower.
    * KNN predicts dataset with features relevant to diabetes (age, blood sugar, etc.) 
    and a target variable representing disease progression.
    **Choosing 'k':**
    The value of 'k' significantly impacts KNN performance. A small 'k' value might lead to overfitting, where the 
    model performs well on the training data but poorly on unseen data. Conversely, a large 'k' value might not 
    capture the local patterns in the data and lead to underfitting. The optimal 'k' value is often determined 
    through experimentation.
    \n**Advantages of KNN:**
    * Simple to understand and implement.
    * No complex model training required.
    * Effective for datasets with well-defined clusters."""
    st.write(text)
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # Load the Diabetes dataset
        diabetes = datasets.load_diabetes()
        X = diabetes.data  # Features
        y = diabetes.target  # Target labels (species)

        # KNN for supervised classification (reference for comparison)

        # Define the KNN classifier with k=5 neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the KNN model
        knn.fit(X, y)

        # Predict the cluster labels for the data
        y_pred = knn.predict(X)
        st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)
        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X[indices, 0], X[indices, 1], label=diabetes.target_names[label], c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Sepal length (cm)')
        ax.set_ylabel('Sepal width (cm)')
        ax.set_title('Sepal Length vs Width Colored by Predicted Diabetes Species')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.title("📊 Train/Test Split App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload trainLabels.csv", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.write(df.head())

    # Encode labels
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    # Select split ratio
    test_size = st.slider("Select Test Size (percentage)", 10, 50, 20, step=5) / 100

    # Train/Test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label_encoded"]
    )

    st.success(f"✅ Data split done: {len(train_df)} train samples , {len(test_df)} test samples")

    # Show class distribution
    st.subheader("Class Distribution in Train/Test")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Train Set Distribution")
        st.write(train_df["label"].value_counts())

    with col2:
        st.write("Test Set Distribution")
        st.write(test_df["label"].value_counts())

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8,4))
    train_df["label"].value_counts().plot(kind="bar", alpha=0.7, color="blue", label="Train", ax=ax)
    test_df["label"].value_counts().plot(kind="bar", alpha=0.7, color="orange", label="Test", ax=ax)
    plt.legend()
    plt.title("Class Distribution in Train vs Test")
    st.pyplot(fig)

    # Download buttons
    st.subheader("Download Splits")
    st.download_button(
        label="Download Train CSV",
        data=train_df.to_csv(index=False),
        file_name="train_split.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Test CSV",
        data=test_df.to_csv(index=False),
        file_name="test_split.csv",
        mime="text/csv"
    )

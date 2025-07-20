import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🌤️ Weather Prediction - ML Model Deployment")

# Step 1: Upload Dataset
st.header("Step 1: Upload Your Weather Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully!")
    st.write("📊 Dataset Preview:")
    st.dataframe(df.head())

    # Step 2: Select Features
    st.header("Step 2: Train ML Model")
    target_col = st.selectbox("Select the column to Predict (Target Variable)", df.columns)

    features = st.multiselect("Select Feature Columns (Independent Variables)", 
                              [col for col in df.columns if col != target_col])

    if features and target_col:
        X = df[features]
        y = df[target_col]

        # Train/Test Split and Model Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model
        with open("weather_model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.success("✅ Model Trained Successfully!")
        st.write("📈 Training Score:", model.score(X_train, y_train))
        st.write("📉 Testing Score:", model.score(X_test, y_test))

        # Step 3: Prediction UI
        st.header("Step 3: Try Prediction")

        input_values = []
        for col in features:
            val = st.number_input(f"Enter value for {col}", value=0.0)
            input_values.append(val)

        if st.button("Predict"):
            model_input = [input_values]
            prediction = model.predict(model_input)
            st.success(f"🌡️ Predicted {target_col}: {prediction[0]:.2f}")

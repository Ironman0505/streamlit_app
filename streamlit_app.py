# linear_regression_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample data generation
import numpy as np

np.random.seed(42)
cgpa = np.random.uniform(6, 9, 200)
iq = np.random.uniform(80, 150, 200)
package = 5000 + 800 * cgpa + 50 * iq + np.random.normal(0, 300, 200)
data = pd.DataFrame({'CGPA': cgpa, 'IQ': iq, 'Package': package})

# Streamlit app
def main():
    st.title("Linear Regression App")

    # Display sample data
    st.subheader("Sample Data:")
    st.dataframe(data.head())

    # Scatter plot of CGPA vs Package
    st.subheader("Scatter Plot:")
    plt.scatter(data['CGPA'], data['Package'])
    plt.xlabel('CGPA')
    plt.ylabel('Package')
    st.pyplot(plt)

    # Scatter plot of IQ vs Package
    st.subheader("Scatter Plot:")
    plt.scatter(data['IQ'], data['Package'])
    plt.xlabel('IQ')
    plt.ylabel('Package')
    st.pyplot(plt)

    # Linear Regression model
    st.subheader("Linear Regression Model:")

    # Select features and target variable
    X = data[['CGPA', 'IQ']]
    y = data['Package']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)

    st.write("Mean Squared Error:", mse)

    # User input for prediction
    st.subheader("Predict Package:")
    cgpa_input = st.slider("Select CGPA:", float(data['CGPA'].min()), float(data['CGPA'].max()))
    iq_input = st.slider("Select IQ:", float(data['IQ'].min()), float(data['IQ'].max()))

    # Make a prediction
    prediction = model.predict([[cgpa_input, iq_input]])

    st.write(f"Predicted Package: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()

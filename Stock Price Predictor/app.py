import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Title
st.title("Stock Price Graph Predictor Using Keras")

# User Input
stock = st.text_input("Enter the Stock ID", "GOOG")

# Fetch Stock Data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

# Debugging: Check columns
st.write("Columns in google_data:", google_data.columns)

# Drop missing values to prevent errors
google_data = google_data.dropna()

# Load Model
model = load_model("stock_price_model.keras")

# Display Data
st.subheader("Stock Data")
st.write(google_data)

# Splitting Data
splitting_len = int(len(google_data) * 0.7)
x_test = google_data.iloc[splitting_len:][["Close"]]  # Fixed indexing

# Plot Function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, "Orange")
    plt.plot(full_data["Close"], "b")
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Moving Averages
google_data["MA_200_days"] = google_data["Close"].rolling(200).mean()
google_data["MA_100_days"] = google_data["Close"].rolling(100).mean()

st.subheader("Original Close Price & Mean Avg 200 Days")
st.pyplot(plot_graph((15,5), google_data["MA_200_days"], google_data))

st.subheader("Original Close Price & Mean Avg 100 Days")
st.pyplot(plot_graph((15,5), google_data["MA_100_days"], google_data))

st.subheader("Original Close Price & Mean Avg 200 & 100 Days")
st.pyplot(plot_graph((15,5), google_data["MA_100_days"], google_data, 1, google_data["MA_200_days"]))

# Scaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)  # No need for [["Close"]], already a DataFrame

# Creating Sequences
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)  # Fixed numpy conversion

# Predicting
predictions = model.predict(x_data)

# Inverse Transform
inverse_pred = scaler.inverse_transform(predictions)
inverse_y_test = scaler.inverse_transform(y_data)

# DataFrame for Plotting
ploting_data = pd.DataFrame(
    {
        "Original_Test_Data": inverse_y_test.reshape(-1),
        "Predicted_Test_Data": inverse_pred.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

# Display Predictions
st.subheader("Original Data vs Predicted Data")
st.write(ploting_data)

# Final Graph
st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15,5))
plt.plot(pd.concat([google_data["Close"][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data unused", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)

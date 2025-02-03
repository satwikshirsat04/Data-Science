import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


st.title("Stock Price Graph Predictor Using Keras")

stock = st.text_input("Enter the Stock ID", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

google_data = google_data.dropna()

model = load_model("stock_price_model.keras")

st.subheader("Stock Data")
st.write(google_data)

# Splitting Data
splitting_len = int(len(google_data) * 0.7)
x_test = google_data.iloc[splitting_len:][["Close"]]

# Scaling Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

# Inverse Transform
inverse_pred = scaler.inverse_transform(predictions)
inverse_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        "Original_Test_Data": inverse_y_test.reshape(-1),
        "Predicted_Test_Data": inverse_pred.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15,5))
plt.plot(pd.concat([google_data["Close"][:splitting_len+100], ploting_data], axis=0))
plt.legend(["Training Data", "Original Test Data", "Predicted Test Data"])
st.pyplot(fig)

import numpy as np
import pandas as pd

# Predict Next 30 Business Days (Excluding Weekends)
future_days = 30
last_150_days = scaled_data[-150:].reshape(1, 150, 1)  # Ensure correct shape
future_predictions = []

future_dates = []
current_date = google_data.index[-1]

while len(future_dates) < future_days:
    current_date += pd.Timedelta(days=1)
    if current_date.weekday() < 5:  # Monday to Friday only
        future_dates.append(current_date)

for _ in range(future_days):
    next_pred = model.predict(last_150_days)[0, 0]
    future_predictions.append(next_pred)

    last_150_days = np.append(last_150_days[:, 1:, :], [[[next_pred]]], axis=1)


future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_df = pd.DataFrame({"Predicted_Close": future_predictions.flatten()}, index=future_dates)

# print(future_df.head())



# Plot Last 70 Days & Next 30 Days Predictions Linear Style
st.subheader(f"Stock Price Prediction: Next 30 Days ({stock})")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(google_data.index[-150:], google_data["Close"].values[-150:], color="green", label="Last 150 Days Actual Price")
ax.plot(future_df.index, future_df["Predicted_Close"], color="red", label="Predicted Next 30 Days")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.set_title(f"Stock Price Prediction for {stock} (Next 30 Days)")
ax.grid(True)
ax.legend()

st.pyplot(fig)
st.subheader("Predicted Stock Prices for Next 30 Days")
st.write(future_df)

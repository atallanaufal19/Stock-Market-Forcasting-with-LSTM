import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn  # Tambahkan ini
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import LSTM, TimeSeriesDataset, train_one_epoch, validate_one_epoch
from main import prepare_dataframe_for_lstm, simulate_gbm, predict_future_drift
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import numpy as np

# Streamlit app
st.title('Stock Market Prediction')
st.write("Masukkan kode perusahaan dan jumlah hari untuk memprediksi harga saham.")

company_code = st.text_input("Kode Perusahaan", value="TLKM.JK")
steps = st.number_input("Jumlah Hari untuk Prediksi", min_value=1, max_value=365, value=30)

if st.button("Show Current Stock Market"):
    data = yf.download(company_code, start="2019-05-01", end="2024-06-10")
    st.line_chart(data['Close'])
    st.write(data.tail())

if st.button("Predict Future Prices"):
    data = yf.download(company_code, start="2019-05-01", end="2024-06-10")
    data = data[['Close']].sort_index(ascending=False)
    
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    
    X = dc(np.flip(X, axis=1))
    
    split_index = int(len(X) * 0.95)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LSTM(1, 4, 1)
    model.to(device)
    
    learning_rate = 0.001
    num_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, loss_function, optimizer)
        val_loss = validate_one_epoch(model, test_loader, loss_function)
        st.write(f'Epoch {epoch + 1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}')
    
    initial_input = X_test[-1].unsqueeze(0)
    future_drift = predict_future_drift(model, initial_input, steps, device)
    
    S0 = data['Close'].iloc[0]
    mu = np.mean(future_drift)
    sigma = np.std(future_drift)
    
    simulated_prices = simulate_gbm(S0, mu, sigma, steps, 10)
    
    plt.figure(figsize=(14, 7))
    plt.plot(simulated_prices, color='grey', alpha=0.5)
    plt.title(f'Simulasi GBM untuk Harga Saham {company_code}')
    plt.xlabel("Hari")
    plt.ylabel("Harga")
    st.pyplot(plt)

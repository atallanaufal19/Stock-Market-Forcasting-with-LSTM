import streamlit as st
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from torch.utils.data import Dataset, DataLoader

# Define the LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define function to prepare data for LSTM
def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# Define function to simulate GBM
def simulate_gbm(S0, mu, sigma, steps, n_simulations):
    dt = 1 / steps
    prices = np.zeros((steps, n_simulations))
    prices[0] = S0
    for t in range(1, steps):
        Z = np.random.standard_normal(n_simulations)
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

# Define function to predict future drift using LSTM model
def predict_future_drift(model, initial_input, steps):
    model.eval()
    predictions = []
    current_input = initial_input.clone().to(device)
    for _ in range(steps):
        with torch.no_grad():
            predicted_price = model(current_input).item()
            predictions.append(predicted_price)
            new_input = torch.tensor(predicted_price).unsqueeze(0).unsqueeze(1).to(device)
            current_input = torch.cat((current_input[:, 0:], new_input.unsqueeze(0)), dim=1)
    return predictions

# Define TimeSeriesDataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Streamlit app
st.title('Stock Market Prediction')
st.write("Masukkan kode perusahaan dan jumlah hari untuk memprediksi harga saham.")

company_code = st.text_input("Kode Perusahaan", value="TLKM.JK")
steps = st.number_input("Jumlah Hari untuk Prediksi", min_value=1, max_value=365, value=30)

if st.button("Show Current Stock Market"):
    data = yf.download(company_code, start="2014-06-20", end="2024-06-20")
    st.line_chart(data['Close'].sort_index(ascending=False))
    st.write(data.tail())

if st.button("Predict Future Prices"):
    data = yf.download(company_code, start="2014-06-20", end="2024-06-20")
    data = data[['Close']].sort_index(ascending=False)
    
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(data, lookback)
    
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    
    X = dc(np.flip(X, axis=1))
    
    split_index = int(len(X) * 0.8)
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
    num_epochs = 20
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def train_one_epoch():
        model.train(True)
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return running_loss / len(train_loader)
    
    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()
        return running_loss / len(test_loader)
    
    for epoch in range(num_epochs):
        train_loss = train_one_epoch()
        val_loss = validate_one_epoch()
        st.write(f'Epoch {epoch + 1}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}')
    
    initial_input = X_test[-1].unsqueeze(0)
    future_drift = predict_future_drift(model, initial_input, steps)
    
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

# Run the Streamlit app
if __name__ == '__main__':
    st.write("Silakan masukkan kode perusahaan dan jumlah hari untuk prediksi.")

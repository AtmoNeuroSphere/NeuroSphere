import math
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def generate_sp500_data(years=10):
    """Generate synthetic S&P 500 data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    dates = pd.date_range(start_date, end_date, freq='B')  # Business days
    
    # Generate synthetic log returns (daily returns in log space)
    n_days = len(dates)
    log_returns = np.random.normal(0.0003, 0.01, n_days)  # Mean return ~0.03% daily, 1% std dev
    
    # Create price series (in log space, then exp to get actual prices)
    log_prices = np.cumsum(log_returns) + np.log(4000)  # Start around 4000
    prices = np.exp(log_prices)
    
    # Add some volatility clustering
    for _ in range(5):
        idx = np.random.randint(0, n_days - 10, n_days // 20)
        for i in idx:
            log_returns[i:i+10] *= 1.5  # Increase volatility in some periods
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates[:len(prices)],
        'Open': prices * (1 + np.random.normal(0, 0.001, len(prices))),
        'High': prices * (1 + np.abs(np.random.normal(0.002, 0.002, len(prices)))),
        'Low': prices * (1 - np.abs(np.random.normal(0.002, 0.002, len(prices)))),
        'Close': prices,
        'Volume': np.random.lognormal(20, 1, len(prices)).astype(int)
    })
    
    # Ensure High >= Open/Close >= Low
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(np.random.normal(0, 0.001, len(df)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(np.random.normal(0, 0.001, len(df)))
    
    return df

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int = 60):
        self.X, self.y, self.dates = [], [], []
        
        # Calculate features
        df["log_close"] = np.log(df["Close"])
        df["ret"] = df["log_close"].diff()
        df = df.dropna()
        df["vol20"] = df["ret"].rolling(20).std().bfill()
        df["ma20"] = df["log_close"].rolling(20).mean().bfill()
        df["ma60"] = df["log_close"].rolling(60).mean().bfill()
        df["trend20"] = df["ma20"] - df["ma60"]
        
        # Prepare features and target
        feats = df[["ret", "vol20", "trend20"]].values.astype("float32")
        target = df["ret"].shift(-1).values.astype("float32")
        
        # Create sequences
        for i in range(len(df) - lookback - 1):
            self.X.append(feats[i:i+lookback])
            self.y.append(target[i+lookback])
            self.dates.append(df["Date"].iloc[i+lookback])
            
        self.X = np.stack(self.X)
        self.y = np.array(self.y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float().to(device),
            torch.tensor(self.y[idx], dtype=torch.float32).to(device),
            str(self.dates[idx])
        )

class LSTMModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        return self.linear(last_output).squeeze(-1)

def train_model():
    # Generate synthetic data
    print("Generating synthetic stock data...")
    df = generate_sp500_data(years=5)
    
    # Create dataset
    lookback = 60
    dataset = StockDataset(df, lookback=lookback)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        # Calculate average losses
        train_loss = np.sqrt(train_loss / len(train_loader.dataset))
        val_loss = np.sqrt(val_loss / len(val_loader.dataset))
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train RMSE: {train_loss:.6f}, "
              f"Val RMSE: {val_loss:.6f}")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    model = train_model()
    # Save the model
    torch.save(model.state_dict(), "stock_predictor.pth")
    print("Model saved as 'stock_predictor.pth'")

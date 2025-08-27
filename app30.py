import math, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"

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

def load_series(csv_path: str = None) -> pd.DataFrame:
    """Load data from CSV or generate synthetic data if file not found."""
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        print("CSV file not found. Generating synthetic S&P 500 data...")
        df = generate_sp500_data()
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_close"] = np.log(df["Close"])
    df["ret"] = df["log_close"].diff()
    df = df.dropna().reset_index(drop=True)
    df["vol20"] = df["ret"].rolling(20).std().bfill()
    df["ma20"] = df["log_close"].rolling(20).mean().bfill()
    df["ma60"] = df["log_close"].rolling(60).mean().bfill()
    df["trend20"] = df["ma20"] - df["ma60"]
    return df

class SeqDS(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int = 60):
        self.X, self.y, self.dates = [], [], []
        feats = df[["ret", "vol20", "trend20"]].values.astype("float32")
        y = df["ret"].shift(-1).values.astype("float32")
        for i in range(len(df)-lookback-1):
            self.X.append(feats[i:i+lookback])
            self.y.append([i+lookback])
            self.dates.append(df["Date"].iloc[i+lookback])
        self.X = np.stack(self.X)
        self.y = np.array(self.y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i]), str(self.dates[i])

class LSTMHead(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        out,_ = self.lstm(x)
        h =out[:,-1,:]
        return self.head(h).squeeze(-1)

@dataclass
class TrainCfg:
    epochs: int=20
    bs: int = 128
    lr: float = 1e-3
    lookback: int = 60
    val_split: float = 0.2

def split_dataset(ds: SeqDS, val_split: float=0.2):
    n = len(ds)  # Use the dataset length instead of undefined df
    n_val = int(n * val_split)
    idx = np.arange(n)
    train_idx, val_idx = idx[:-n_val], idx[-n_val:]
    subset = lambda ii: torch.utils.data.Subset(ds, ii.tolist())
    return subset(train_idx), subset(val_idx)

def rmse(a,b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))

def train(csv_path="sp500.csv", model_out="sp500_lstm.pt", cfg=TrainCfg()):
    df = load_series(csv_path)
    ds = SeqDS(df, cfg.lookback)
    train_ds, val_ds = split_dataset(ds, cfg.val_split)
    train_dl = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False)

    model = LSTMHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()

    best = math.inf
    for ep in range(cfg.epochs):
        model.train()
        losses=[]
        for X,y,_ in train_dl:
            X = X.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        preds=[]; gts=[]
        with torch.no_grad():
            for X,y,_ in val_dl:
                X=X.to(DEVICE); y=y.to(DEVICE)
                p = model(X)
                preds.append(p.cpu().numpy());gts.append(y.cpu().numpy())
        preds = np.concatenate(preds); gts = np.concatenate(gts)
    val_rmse = rmse(preds, gts)
    print(f"epoch {ep+1}/{cfg.epochs} train={np.mean(losses):.4f} val_rmse={val_rmse:.6f}")
    if val_rmse < best:
        best = val_rmse
        torch.save({"state_dict":model.state_dict(),"cfg":cfg.__dict__}, model_out)
    return model_out, best

def anomlay_score(lastest_seq: np.ndarray, reported_next_ret: float, model_path="sp500lstm.pt") -> float:
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = LSTMHead().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    X = torch.from_numpy(lastest_seq.astype("float32")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(X).item()
    resid = abs(pred - float(reported_next_ret))
    return 1e4 * resid

def tracking_error_bps(preds: np.ndarray, realized: np.ndarray) -> float:
    reside = preds - realized
    te = np.std(resid) * np.sqrt(252) * 1e4
    return float(te)

if __name__ == "__main__":
    path, best = train()
    print(f"Saved:", path, "best_val_rmse:", best)

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import os, time, jwt

from sp500_tracker import anomlay_score

JWT_AUDIENCE = os.getenv("JWT_AUD", "cbdc-backend")
JWT_ISSUER = os.getenv("JWT_ISS", "auth.bank.example")
JWT_KEY = os.getenv("JTW_KEY", "dev-only-key")

app = FastAPI(title="CBDC Risk/ML & Peg API")

def verify_jwt(auth: str):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = auth.split()[1]
    try:
        claims = jwt.decode(token, JWT_KEY, algorithms=["HS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        return claims
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

class ScoreIn(BaseModel):
    latest_seq: list
    reported_next_ret: float

@app.middleware("http")
async def strict_security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return resp

@app.post("/ml/anomaly-score")
def ml_score(payload: ScoreIn, authorization: str=Header(None)):
    verify_jwt(authorization)
    arr = np.array(payload.latest_seq, dtype="float32")
    score = anomlay_score(arr, payload.reported_next_ret)  # Fixed function name
    return {"anomaly_score_bps": score}

@app.get("/peg/value")
def peg_value(index_t: float, index_0: float, v0: float = 1.0, authorization: str = Header(None)):
    verify_jwt(authorization)
    return {"value": v0 * (index_t / index_0)}

# YAML configuration for Envoy proxy (commented out to avoid syntax errors)
# static_resources:
#     listeners:
#     - name: https_listener
#       address: { socket_address: { address: 0.0.0.0, port_value: 443 }}
#       filter_chains:
#       - transport_socket:
#           name: envoy.transport_sockets.tls
#           typed_config:
#             "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
#             common_tls_context:
#               tls_certificates:
#               - certificate_chain: {filename: "/etc/tls/server.crt"}
#                 private_key: {filename: "/etc/tls/server.key"}
#               validation_context:
#                 trusted_ca: { filename: "/etc/tls/ca.crt" }
#                 require_client_certificate: true
# 
#     filters:
#     - name: envoy.filters.network.http_connection_manager
#       typed_config:
#         "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
#         stat_prefix: ingress_http
#         http_filters:
#         - name: envoy.filters.http.router
#         route_config:
#           name: local_route
#           virtual_hosts:
#           - name: backend
#             domains: ["*"]
#             routes:
#             - match: { prefix: "/ml" }
#               route: { cluster: risk_service }
#               request_headers_to_remove: ["Cookie", "Referer", "X-Forwarded-For", "X-Real-IP", "User-Agent"]
#             - match: { prefix: "/peg" }
#               route: { cluster: risk_service }
# 
# clusters:
# - name: risk_service
#   connect_timeout: 0.25s
#   type: STRICT_DNS
#   lb_policy: ROUND_ROBIN
#   load_assignment:
#     cluster_name: risk_service
#     endpoints:
#     - lb_endpoints:
#       - endpoint:
#           address:
#             socket_address:
#               address: cbdc-ml
#               port_value: 8000
#   transport_socket:
#     name: envoy.transport_sockets.tls
#     typed_config:
#       "@type": type.googleapis.com/envoy.extensions.transport_socket.tls.v3.UpstreamTlsContext
#       common_tls_context:
#         tls_certificates:
#         - certificate_chain: {filename: "/etc/tls/client.crt"}
#           private_key: {filename: "/etc/tls/client.key"}
#         validation_context:
#           trusted_ca: { filename: "/etc/tls/ca.crt" }
      
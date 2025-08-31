# src/train.py
from pathlib import Path
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- paths
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

# --- load data
train_df = pd.read_csv(DATA / "labelled_train.csv")
test_df  = pd.read_csv(DATA / "labelled_test.csv")
val_df   = pd.read_csv(DATA / "labelled_validation.csv")

X_train = train_df.drop("sus_label", axis=1).values
y_train = train_df["sus_label"].astype("float32").values
X_test  = test_df.drop("sus_label", axis=1).values
y_test  = test_df["sus_label"].astype("float32").values
X_val   = val_df.drop("sus_label", axis=1).values
y_val   = val_df["sus_label"].astype("float32").values

# --- scale + save scaler
scaler = StandardScaler().fit(X_train)
X_train, X_test, X_val = scaler.transform(X_train), scaler.transform(X_test), scaler.transform(X_val)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# --- tensors
Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
Xte = torch.tensor(X_test,  dtype=torch.float32)
yte = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)
Xva = torch.tensor(X_val,   dtype=torch.float32)
yva = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)

# --- model (binary) + loss/opt
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1)  # no sigmoid here
)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --- train
for _ in range(20):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xtr), ytr)
    loss.backward()
    optimizer.step()

# --- eval
model.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(model(Xtr)).numpy().ravel()
    p_va = torch.sigmoid(model(Xva)).numpy().ravel()
    p_te = torch.sigmoid(model(Xte)).numpy().ravel()

th = 0.5
print(f"Training accuracy: {accuracy_score(y_train, (p_tr>=th).astype(int)):.3f}")
print(f"Validation accuracy: {accuracy_score(y_val,   (p_va>=th).astype(int)):.3f}")
print(f"Testing accuracy: {accuracy_score(y_test,   (p_te>=th).astype(int)):.3f}")

# --- save model weights
torch.save(model.state_dict(), MODEL_DIR / "model.pt")

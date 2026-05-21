
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# 1. 재현성 고정
torch.manual_seed(42)
np.random.seed(42)

# 2. 예제 시계열 데이터 생성 :  사인파 + 약간의 노이즈
time_steps = 1000
x = np.arange(0, time_steps, dtype=np.float32)
series = np.sin(0.02 * x) + 0.1 * np.random.randn(time_steps).astype(np.float32)

# 3. 정규화 :  LSTM은 입력 스케일이 안정적일수록 학습이 잘 됨
mean = series.mean()
std = series.std()
series = (series - mean) / (std + 1e-8)

# 4. 슬라이딩 윈도우 데이터셋 생성 :  과거 20개 시점으로 다음 1개 예측
sequence_length = 20

def create_dataset(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x_seq = data[i:i + seq_len]
        y_seq = data[i + seq_len]
        xs.append(x_seq)
        ys.append(y_seq)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X, y = create_dataset(series, sequence_length)

# 샘플데이터(상승하는 자료배열) 출력해보기
sample = X[0].squeeze()  # (20,)
plt.plot(sample)
plt.title("One input sequence")
plt.show()

# LSTM 입력 형태: (batch, seq_len, input_size)
X = np.expand_dims(X, axis=-1)  # (N, 20, 1)
y = np.expand_dims(y, axis=-1)  # (N, 1)

# 5. 학습/검증 데이터 분리
train_size = int(len(X) * 0.8)

X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 6. LSTM 모델 정의
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # num_layers >= 2일 때만 의미 있음
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 가장 마지막 시점 출력 사용
        # lstm_out[:, -1, :] shape: (batch, hidden_size)
        out = self.fc(lstm_out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. 학습 함수
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)

# 8. 학습 실행
epochs = 30

for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    if epoch % 5 == 0 or epoch == 1:
        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# 9. 예측 확인
# model.eval()
# with torch.no_grad():
#     sample_x = X_val[:5].to(device)   # 검증 데이터 5개
#     sample_pred = model(sample_x).cpu().numpy()
#     sample_true = y_val[:5].cpu().numpy()

# print("\n예측값 vs 실제값")
# for i in range(5):
#     print(f"Pred: {sample_pred[i][0]:.4f}, True: {sample_true[i][0]:.4f}")



model.eval()
with torch.no_grad():
    preds = model(X_val.to(device)).cpu().numpy()
    true = y_val.cpu().numpy()

plt.figure(figsize=(12, 5))
plt.plot(true[:200], label='True')
plt.plot(preds[:200], label='Pred')
plt.legend()
plt.title("LSTM Prediction vs True")
plt.show()
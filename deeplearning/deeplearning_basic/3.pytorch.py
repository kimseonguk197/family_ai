import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# -----------------------------
# 1) 시드 고정
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# 2) 데이터 생성
# -----------------------------
x_np = np.random.rand(1000, 1).astype(np.float32)
noise = np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)
y_np = 3.0 * x_np + 2.0 + noise

# NumPy -> PyTorch Tensor 변환
x = torch.tensor(x_np)
y = torch.tensor(y_np)

# -----------------------------
# 3) Dataset / DataLoader 구성
#    TensorDataset: 텐서를 데이터셋처럼 묶어줌
#    DataLoader: 배치 처리, 셔플 담당
# -----------------------------
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 4) 모델 정의
#    nn.Module을 상속받아 모델 구조 정의
# -----------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 1개 -> 출력 1개 선형층
        # 내부적으로 weight와 bias를 가짐
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        # forward는 입력이 들어왔을 때 계산 흐름 정의
        return self.linear(x)

model = LinearRegressionModel()

# -----------------------------
# 5) 손실 함수와 옵티마이저 정의
# -----------------------------
criterion = nn.MSELoss()                      # 평균 제곱 오차
optimizer = optim.Adam(model.parameters(), lr=0.1)

# -----------------------------
# 6) 학습 루프
# -----------------------------
epochs = 10

for epoch in range(epochs):
    epoch_loss = 0.0

    for batch_x, batch_y in dataloader:
        # 1. 이전 step의 gradient 초기화
        optimizer.zero_grad()

        # 2. 순전파(forward)
        pred = model(batch_x)

        # 3. 손실 계산
        loss = criterion(pred, batch_y)

        # 4. 역전파(backward)
        loss.backward()

        # 5. 파라미터 업데이트
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1:02d}, Loss: {avg_loss:.4f}")

# -----------------------------
# 7) 학습된 파라미터 확인
# -----------------------------
weight = model.linear.weight.data
bias = model.linear.bias.data

print("\n학습된 파라미터")
print("weight =", weight)
print("bias =", bias)

# -----------------------------
# 8) 예측
# -----------------------------
test_x = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)

# 추론 시에는 gradient 계산이 불필요하므로 no_grad 사용
with torch.no_grad():
    pred = model(test_x)

print("\n예측 결과")
for inp, out in zip(test_x, pred):
    print(f"x={inp[0].item():.1f} -> y_pred={out[0].item():.4f}")
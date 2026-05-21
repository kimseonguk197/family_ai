import tensorflow as tf
import numpy as np

# -----------------------------
# 1) 시드 고정
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)

# -----------------------------
# 2) 데이터 생성
# -----------------------------
x = np.random.rand(1000, 1).astype(np.float32)
noise = np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)
y = 3.0 * x + 2.0 + noise

# -----------------------------
# 3) Keras 모델 정의
#    Sequential: 층을 순서대로 쌓는 가장 단순한 형태
# -----------------------------
model = tf.keras.Sequential([
    # 입력 1개를 받아 출력 1개를 내는 Dense layer
    # 사실상 y = wx + b 와 동일한 선형 회귀 모델
    tf.keras.layers.Dense(units=1, input_shape=(1,))
])

# -----------------------------
# 4) 모델 컴파일
#    optimizer: 파라미터 업데이트 방식
#    loss: 손실 함수
#    metrics: 모니터링용 지표
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mse',
    metrics=['mae']
)

# -----------------------------
# 5) 모델 학습
#    epochs: 전체 데이터를 몇 번 반복할지
#    batch_size: 한 번에 몇 개씩 학습할지
#    validation_split: 일부 데이터를 검증용으로 분리
# -----------------------------
history = model.fit(
    x, y,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 6) 모델 평가
# -----------------------------
loss, mae = model.evaluate(x, y, verbose=0)
print(f"\n최종 평가 - Loss(MSE): {loss:.4f}, MAE: {mae:.4f}")

# -----------------------------
# 7) 학습된 가중치 확인
#    Dense layer의 weight와 bias를 확인
# -----------------------------
weights, bias = model.layers[0].get_weights()
print("\n학습된 파라미터")
print("weight =", weights)
print("bias =", bias)

# -----------------------------
# 8) 예측
# -----------------------------
test_x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
pred = model.predict(test_x)

print("\n예측 결과")
for inp, out in zip(test_x, pred):
    print(f"x={inp[0]:.1f} -> y_pred={out[0]:.4f}")
import tensorflow as tf
import numpy as np

# -----------------------------
# 1) 재현 가능성을 위한 시드 고정
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)

# -----------------------------
# 2) 학습용 데이터 생성
#    실제 정답은 y = 3x + 2 + noise
# -----------------------------
x_np = np.random.rand(1000, 1).astype(np.float32)  # 0~1 사이의 입력 1000개
noise = np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)
y_np = 3.0 * x_np + 2.0 + noise

# TensorFlow Tensor로 변환
x = tf.constant(x_np)
y = tf.constant(y_np)

# -----------------------------
# 3) 학습 가능한 파라미터 정의
#    Variable은 학습 중 값이 바뀌는 변수
# -----------------------------
w = tf.Variable(tf.random.normal(shape=(1, 1)))  # 가중치
b = tf.Variable(tf.zeros(shape=(1,)))            # 편향

# -----------------------------
# 4) 예측 함수 정의
#    y_pred = xw + b
# -----------------------------
def predict(x):
    return tf.matmul(x, w) + b

# -----------------------------
# 5) 손실 함수 정의
#    회귀에서는 MSE(평균 제곱 오차)를 많이 사용
# -----------------------------
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# -----------------------------
# 6) Optimizer 정의
#    Adam은 실무/학습에서 매우 자주 사용
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# -----------------------------
# 7) Dataset 구성
#    tf.data.Dataset은 TensorFlow에서 배치 처리의 표준
# -----------------------------
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1000).batch(32)

# -----------------------------
# 8) 한 번의 학습 step 정의
#    GradientTape로 자동 미분 수행
# -----------------------------
def train_step(batch_x, batch_y):
    # GradientTape는 forward 연산을 기록해서 backward 계산에 사용
    with tf.GradientTape() as tape:
        # 예측값 계산
        y_pred = predict(batch_x)
        # 손실 계산
        loss = loss_fn(batch_y, y_pred)

    # w, b에 대한 손실의 기울기 계산
    gradients = tape.gradient(loss, [w, b])

    # optimizer가 기울기를 사용해서 파라미터를 업데이트
    optimizer.apply_gradients(zip(gradients, [w, b]))

    return loss

# -----------------------------
# 9) 전체 학습 루프
# -----------------------------
epochs = 10

for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0

    for batch_x, batch_y in dataset:
        batch_loss = train_step(batch_x, batch_y)
        epoch_loss += batch_loss.numpy()
        batch_count += 1

    print(f"Epoch {epoch+1:02d}, Loss: {epoch_loss / batch_count:.4f}")

# -----------------------------
# 10) 학습 결과 확인
#    실제 정답은 w ≈ 3, b ≈ 2
# -----------------------------
print("\n학습된 파라미터")
print("w =", w.numpy())
print("b =", b.numpy())

# -----------------------------
# 11) 새 데이터 예측
# -----------------------------
test_x = tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float32)
test_pred = predict(test_x)

print("\n예측 결과")
for inp, pred in zip(test_x.numpy(), test_pred.numpy()):
    print(f"x={inp[0]:.1f} -> y_pred={pred[0]:.4f}")
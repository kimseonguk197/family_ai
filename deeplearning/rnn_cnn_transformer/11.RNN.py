import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN
from tensorflow.keras.optimizers import Adam

# 재현성
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# 1. 데이터 생성
def generate_order_dataset(n_samples=10000, seq_len=8, vocab_size=10):
    """
    label=1 : token 1 이 token 2 보다 먼저 등장
    label=0 : token 2 가 token 1 보다 먼저 등장
    나머지 위치는 3  사이의 랜덤 토큰으로 채움  -> 전체 토큰 개수는 비슷하지만, 정답은 순서에 의해 결정됨
    """
    X_seq = []
    y = []

    for _ in range(n_samples):
        seq = np.random.randint(3, vocab_size, size=seq_len)

        # 1과 2가 들어갈 서로 다른 위치 2개 선택
        pos1, pos2 = np.random.choice(seq_len, size=2, replace=False)

        # 절반은 1이 먼저, 절반은 2가 먼저
        label = np.random.randint(0, 2)

        if label == 1:
            # 1이 먼저, 2가 나중
            if pos1 < pos2:
                seq[pos1] = 1
                seq[pos2] = 2
            else:
                seq[pos2] = 1
                seq[pos1] = 2
        else:
            # 2가 먼저, 1이 나중
            if pos1 < pos2:
                seq[pos1] = 2
                seq[pos2] = 1
            else:
                seq[pos2] = 2
                seq[pos1] = 1

        X_seq.append(seq)
        y.append(label)

    return np.array(X_seq), np.array(y)


def seq_to_bow(X_seq, vocab_size=10):
    # 순서 정보를 버리고, 각 토큰이 몇 번 등장했는지만 세는 Bag-of-Words 변환
    X_bow = np.zeros((len(X_seq), vocab_size), dtype=np.float32)
    for i, seq in enumerate(X_seq):
        for token in seq:
            X_bow[i, token] += 1.0
    return X_bow


# 2. 모델 정의
def build_mlp_bow_model(vocab_size=10):
    # Bag-of-Words 벡터를 입력받는 MLP
    model = Sequential([
        Input(shape=(vocab_size,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_rnn_model(vocab_size=10, seq_len=8):
    # 순서가 있는 정수 시퀀스를 입력받는 RNN
    model = Sequential([
        Input(shape=(seq_len,)),
        Embedding(input_dim=vocab_size, output_dim=8),
        SimpleRNN(16),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 3. 메인 실행
if __name__ == "__main__":
    vocab_size = 10
    seq_len = 8

    # 데이터 생성
    X_train_seq, y_train = generate_order_dataset(n_samples=10000, seq_len=seq_len, vocab_size=vocab_size)
    X_test_seq, y_test = generate_order_dataset(n_samples=2000, seq_len=seq_len, vocab_size=vocab_size)

    # MLP용 Bag-of-Words 변환
    X_train_bow = seq_to_bow(X_train_seq, vocab_size=vocab_size)
    X_test_bow = seq_to_bow(X_test_seq, vocab_size=vocab_size)

    print("=== 샘플 데이터 확인 ===")
    for i in range(5):
        print(f"sequence: {X_train_seq[i]}, label: {y_train[i]}, bow: {X_train_bow[i]}")

    # MLP 학습
    print("\n=== MLP(Bag-of-Words) 학습 ===")
    mlp_model = build_mlp_bow_model(vocab_size=vocab_size)
    mlp_model.summary()

    mlp_model.fit(
        X_train_bow, y_train,
        validation_data=(X_test_bow, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    mlp_loss, mlp_acc = mlp_model.evaluate(X_test_bow, y_test, verbose=0)
    print(f"\nMLP 테스트 정확도: {mlp_acc:.4f}")

    # RNN 학습
    print("\n=== RNN 학습 ===")
    rnn_model = build_rnn_model(vocab_size=vocab_size, seq_len=seq_len)
    rnn_model.summary()

    rnn_model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    rnn_loss, rnn_acc = rnn_model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"\nRNN 테스트 정확도: {rnn_acc:.4f}")

    # 예측 비교
    print("\n=== 예측 비교 ===")
    for i in range(10):
        seq = X_test_seq[i]
        bow = X_test_bow[i].reshape(1, -1)

        mlp_pred = (mlp_model.predict(bow, verbose=0)[0][0] > 0.5).astype(int)
        rnn_pred = (rnn_model.predict(seq.reshape(1, -1), verbose=0)[0][0] > 0.5).astype(int)

        print(f"seq={seq}, true={y_test[i]}, MLP={mlp_pred}, RNN={rnn_pred}")
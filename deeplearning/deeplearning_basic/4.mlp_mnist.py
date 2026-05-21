import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical


# 1. MNIST 데이터 로드 및 전처리
def load_mnist_all():
    """
    [전처리 1] 
    MNIST 손글씨 숫자 데이터셋(0~9)을 불러오고(흑백이미지), MLP 입력에 맞게 전처리하는 함수
    - x_train: 학습 이미지 (60000, 28, 28)
    - y_train: 학습 정답 라벨 (60000,) 정답 숫자 라벨(0~9)
    - x_train_flat: MLP 입력을 위해 2차원 이미지(28x28)를 1차원 벡터(784)로 펼침 (60000, 784)
    - y_train_onehot: 원-핫 인코딩된 학습 정답 (60000, 10)
    - x_test: 테스트 이미지 (10000, 28, 28)
    - y_test: 테스트 정답 라벨 (10000,)
    - x_test_flat: 펼친(flatten) 테스트 입력값 (10000, 784)
    - y_test_onehot: 원-핫 인코딩된 테스트 정답 (10000, 10)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # [전처리 2] 픽셀값 정규화  : 원래 MNIST 픽셀값은 0~255 범위인데, 이를 0~1 범위로 바꾸어 학습을 더 안정적으로 만듦
    # 255.0 이 정규화 역할을 함
    x_train_flat = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test_flat = x_test.reshape(-1, 784).astype("float32") / 255.0


    # [전처리 3] 라벨을 원-핫 인코딩
    # 예: 숫자 3 -> [0,0,0,1,0,0,0,0,0,0], 출력층이 10개의 클래스 확률을 내므로, 정답도 길이 10짜리 벡터 형태
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)

    return x_train, y_train, x_train_flat, y_train_onehot, x_test, y_test, x_test_flat, y_test_onehot


# 2. MLP 모델 생성
def build_mlp_model():
    """
    다중 분류용 MLP(Multi-Layer Perceptron) 모델 생성 함수
    모델 구조: 입력층(784) -> 은닉층1(Dense 128, ReLU) -> 은닉층2(Dense 64, ReLU) -> 출력층(Dense 10, Softmax)
    """

    model = Sequential([
        # 입력층: 28x28 이미지를 펼친 784차원 벡터를 입력받음
        Input(shape=(784,)),

        # 은닉층 1
        # Dense(128): 이전 층의 모든 노드와 완전 연결된 128개 뉴런
        # activation="relu": 활성화 함수 ReLU 사용
        # ReLU(x) = max(0, x) :  비선형성을 추가해서 신경망이 복잡한 패턴을 학습할 수 있게 해줌
        Dense(128, activation="relu"),

        # 은닉층 2
        Dense(64, activation="relu"),

        # 출력층
        # Dense(10): 숫자 클래스 0~9, 총 10개 클래스에 대응
        # activation="softmax":  각 클래스에 대한 "확률"처럼 해석 가능한 값을 출력. 10개 출력값의 합은 1
        # 예: [0.01, 0.02, 0.90, ...] 이면 "이 입력은 숫자 2일 확률이 가장 높다"라고 해석
        Dense(10, activation="softmax")
    ])

    # 모델 컴파일
    # optimizer="adam": 가중치를 어떤 방식으로 업데이트할지 결정하는 최적화 알고리즘(경사하강법 기반 알고리즘)
    # loss="categorical_crossentropy": 다중 분류 문제에서 예측 확률과 정답 분포의 차이를 계산하는 손실 함수. 정답이 원-핫 인코딩일 때 사용
    # metrics=["accuracy"]: 학습 중 정확도를 함께 확인
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# 3. 예측 결과 확인 함수 : 테스트셋의 특정 샘플 1개를 예측하고, 실제 정답과 예측 결과를 시각적으로 확인하는 함수
def predict_sample(model, x_test_img, x_test_flat, y_test_real, index=0):
    # MLP는 입력을 (1, 784) 형태로 받아야 하므로 테스트 샘플 1개를 꺼냄
    sample = x_test_flat[index].reshape(1, 784)

    # model.predict(): 순전파(forward propagation)를 수행하여 각 클래스(0~9)에 대한 예측 확률을 계산
    pred_probs = model.predict(sample, verbose=0)[0]

    # 예측 확률이 가장 큰 클래스 인덱스를 최종 예측값으로 선택
    pred_label = np.argmax(pred_probs)

    print(f"테스트셋의 {index + 1}번째 이미지의 실제 숫자값: {y_test_real[index]}")

    # 원본 이미지를 화면에 표시하여 확인하고자 함
    plt.imshow(x_test_img[index], cmap="gray")
    plt.show()

    print(f"모델의 예측 결과: {pred_label}")
    print(f"각 숫자(0~9)에 대한 예측 확률: {np.round(pred_probs, 2)}")


# 4. 메인 실행부
if __name__ == "__main__":

    # (1) 데이터 준비
    x_train_img, y_train_real, x_train, y_train_onehot, \
    x_test_img, y_test_real, x_test, y_test_onehot = load_mnist_all()

    # (2) 모델 생성
    model = build_mlp_model()

    # 모델 구조 출력 :  각 층의 출력 shape, 파라미터 수 등을 확인 가능
    model.summary()

    # (3) 모델 학습
    # model.fit(...) 내부에서 일어나는 핵심 과정:
    # 1) 순전파(Forward Propagation) :  입력 x_train을 모델에 통과시켜 예측값 계산
    # 2) 손실 계산(Loss Calculation) :  예측값과 실제 정답(y_train_onehot)의 차이를 categorical_crossentropy로 계산
    # 3) 역전파(Backpropagation) :  손실이 각 가중치에 얼마나 영향을 주었는지  미분(gradient) 계산
    # 4) 가중치 업데이트(Weight Update):  optimizer=adam이 gradient를 사용해 가중치와 bias를 수정
    # 즉, "역전파"가 이 코드에서는 model.fit(...) 안에서 자동 수행
    model.fit(
        x_train,
        y_train_onehot,
        validation_data=(x_test, y_test_onehot),
        epochs=1,
        batch_size=32
    )

    # (4) 테스트 샘플 1개 예측 : 101번째 데이터는 6임 -> 6일 확률을 출력하여 확인
    predict_sample(model, x_test_img, x_test, y_test_real, index=100)

    # (5) 테스트셋 전체 평가
    loss, acc = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"\n전체 테스트 정확도: {acc:.4f}")

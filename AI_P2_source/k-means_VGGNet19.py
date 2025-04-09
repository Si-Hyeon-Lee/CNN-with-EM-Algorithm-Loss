# EM 알고리즘(k-means)을 통해서 Custom Loss function 생성 후, VGGNet19 모델 사용
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from datetime import date


@tf.function
def label_based_partition(y_true, y_pred):
    labels = tf.argmax(y_true, axis=-1)
    y_pred_partitions = []
    for label in range(10):
        mask = tf.equal(labels, label)
        partition = tf.boolean_mask(y_pred, mask)
        y_pred_partitions.append(partition)
    return y_pred_partitions


@tf.function
def k_means_clustering(data, k=2, max_iterations=100):
    num_points = tf.shape(data)[0]
    dimensions = tf.shape(data)[1]

    centroids = tf.slice(tf.random.shuffle(data), [0, 0], [k, dimensions])

    old_centroids = tf.zeros([k, dimensions])

    def should_continue(i, centroids, old_centroids):
        return i < max_iterations

    def iteration(i, centroids, old_centroids):
        expanded_vectors = tf.expand_dims(data, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)

        distances = tf.reduce_sum(
            tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2
        )
        assignments = tf.argmin(distances, 0)

        means = []
        for c in range(k):
            means.append(
                tf.reduce_mean(
                    tf.gather(
                        data, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])
                    ),
                    axis=[1],
                )
            )

        new_centroids = tf.concat(means, 0)
        return [i + 1, new_centroids, centroids]

    _, centroids, _ = tf.while_loop(
        should_continue, iteration, [0, centroids, old_centroids]
    )

    return centroids


@tf.function
def customloss_by_sihyeonlee(y_true, y_pred):
    y_pred_partitions = label_based_partition(y_true, y_pred)
    sum_similarity = 0
    cent_list = []
    for label_tens in y_pred_partitions:
        cents = k_means_clustering(label_tens)
        label_cent = tf.reduce_sum(cents, axis=0) / 2  # 열끼리 싹 더함.
        cent_list.append(label_cent)  # 리스트에 추가

        cents = tf.nn.l2_normalize(cents, axis=1)  # 코사인 유사도 계산
        similarity = tf.tensordot(cents[0], cents[1], axes=1)
        sum_similarity += tf.reduce_sum(similarity)  # 코사인 유사도 전부 더해서 로그 씌울것.

    cent_concated = tf.stack(cent_list, axis=0)  # 2차원 텐서 생성
    label_distance = tf.norm(
        tf.expand_dims(cent_concated, axis=1) - tf.expand_dims(cent_concated, axis=0),
        axis=-1,
    )
    label_distance = tf.reduce_sum(cent_concated)

    crossentropy = tf.keras.losses.CategoricalCrossentropy()

    return (
        crossentropy(y_true, y_pred)
        - 0.1
        * tf.keras.backend.log(
            1 + sum_similarity
        )  # sum_similarity 는 클 수록 좋은것. 최대 10임. 로스는 나쁜거니까 뺴줘야. 0.1 안해줘도 학습 잘 함.
        - 0.1
        * tf.keras.backend.log(
            1 + label_distance
        )  # label_distance 도 클 수록 좋은것. 라벨과의 거리가 멀어졌다는걸 의미.
    )


regularizer = regularizers.L2(1e-5)

# VGG19 스타일 모델 (배치 정규화 추가) 구축
model = Sequential(
    [
        Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            input_shape=(32, 32, 3),
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
        ),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(1024, kernel_initializer="he_normal"),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(32, kernel_initializer="he_normal"),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(10, activation="softmax", kernel_initializer="he_normal"),
    ]
)
# model check point
today = date.today()
today_str = today.strftime("%Y-%m-%d-%H-%M")
today_weight_path = f"./weights/{today_str}/"

os.makedirs(today_weight_path, exist_ok=True)
file_path = (
    today_weight_path
    + "epochs:{epoch:02d}-val_acc:{val_accuracy:.3f}-val_loss{val_loss:.3f}.hdf5"
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
)

# tensorboard -> $ tensorboard --logdir=./result_log/
log_dir = "./result_log"
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, profile_batch=1
)

# 데이터 셋
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 이미지 정규화
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# 클래스 라벨 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 데이터 증강 (shuffle은 알아서 해줌)
datagen = ImageDataGenerator(
    rotation_range=15,  # 이미지를 최대 15도까지 회전
    width_shift_range=0.1,  # 이미지를 가로로 최대 10%까지 이동
    height_shift_range=0.1,  # 이미지를 세로로 최대 10%까지 이동
    shear_range=0.1,  # 이미지를 최대 10%까지 밀림
    zoom_range=0.1,  # 이미지를 최대 10%까지 확대/축소
    horizontal_flip=True,  # 이미지를 수평으로 뒤집기
    fill_mode="nearest",  # 새롭게 생성된 픽셀을 인접한 픽셀로 채우기
)
datagen.fit(x_train)

# tensorboard -> $ tensorboard --logdir=./result_log/
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=20,
    verbose=1,
    mode="auto",
    mean_delta=0.0001,
    cooldown=5,
    min_lr=1e-7,
)

# Optimizer
adam = tf.keras.optimizers.Adam(learning_rate=0.001)

# 모델 컴파일
model.compile(optimizer=adam, loss=customloss_by_sihyeonlee, metrics=["accuracy"])

# 모델 훈련
model.fit(
    datagen.flow(x_train, y_train, batch_size=512),
    epochs=300,
    validation_data=(x_test, y_test),
    validation_batch_size=512,
    callbacks=[lr_callback, tensorboard_callback, model_checkpoint],
)
model.evaluate(x_test, y_test, batch_size=512)

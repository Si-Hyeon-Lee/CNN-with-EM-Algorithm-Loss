# Bayesian Optimization을 통한 hyper parameter 튜닝
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau
from kerastuner.tuners import BayesianOptimization
import os
from datetime import date

# VGG19 스타일 모델 (배치 정규화 추가) 구축
def create_model(hp):
    model = Sequential([
        # block 1
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal', input_shape=(32, 32, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # block 2
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal', input_shape=(32, 32, 3)),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # block 3
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), depth_multiplier=1, padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # block 4
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), depth_multiplier=1, padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # block 5
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        BatchNormalization(),    
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),

        tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), depth_multiplier=1, padding='same', kernel_regularizer=l2(hp.Float('l2_value', min_value=1e-4, max_value=1e-2, sampling='LOG')), kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Fully connected Layer
        Flatten(),
        Dense(hp.Choice('units', values=[1024, 2048, 4096]), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),
        Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.5)),

        Dense(hp.Choice('units', values=[1024, 2048, 4096]), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(alpha=hp.Float('leaky_relu_alpha', min_value=0.01, max_value=0.3, step=0.01, default=0.01)),
        Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1, default=0.5)),
        
        Dense(10, activation='softmax', kernel_initializer='he_normal')

    ])
    initial_learning_rate = hp.Choice('initial_learning_rate', values=[0.001, 0.01, 0.1])  # tune initial learning rate
    decay_steps = hp.Int('decay_steps', min_value=30000, max_value=70000, step=5000)  # tune decay steps
    cosine_decay = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(cosine_decay), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

today = date.today()
today_str = today.strftime("%Y-%m-%d-%H-%M")
today_weight_path = f"./weights/{today_str}/"

os.makedirs(today_weight_path, exist_ok=True)
file_path = today_weight_path+"epochs:{epoch:02d}-val_acc:{val_accuracy:.3f}-val_loss{val_loss:.3f}.hdf5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto'
)

# tensorboard -> $ tensorboard --logdir=./result_log/
log_dir = './result_log'
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch=1
                                                      )

# 데이터 셋 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 이미지 정규화
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

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
    fill_mode='nearest',  # 새롭게 생성된 픽셀을 인접한 픽셀로 채우기
)
datagen.fit(x_train)


tuner = BayesianOptimization(create_model, 
                             objective='val_accuracy', 
                             max_trials=4, 
                             executions_per_trial=4, 
                             directory=log_dir, 
                             project_name='CIFAR10_hyper')

tuner.search(x_train, y_train, epochs=100, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 최적의 하이퍼파라미터 값 출력
print("Best Hyperparameters:")
print(f"leaky_relu_alpha: {best_hps.get('leaky_relu_alpha')}")
print(f"dropout_rate: {best_hps.get('dropout_rate')}")
print(f"units: {best_hps.get('units')}")
print(f"init_learning_rate: {best_hps.get('initial_learning_rate')}")
print(f"L2_norm_rate : {best_hps.get('l2_value')}")

# 최적의 파라미터로 모델 생성 
model = tuner.hypermodel.build(best_hps)

# 모델 학습
model.fit(datagen.flow(x_train, y_train, batch_size=256),
          epochs=300,
          validation_data=(x_test, y_test),
          callbacks=[model_checkpoint, tensorboard_callback])
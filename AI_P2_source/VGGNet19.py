# VGGNet19
import tensorflow as tf
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
import os
from datetime import date
# VGG19 스타일 모델 (배치 정규화 추가) 구축
model = Sequential([
    Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006), input_shape=(32, 32, 3)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   

    Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),    

    Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),

    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),

    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),

    Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0006)),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Flatten(),
    Dense(1024, kernel_initializer='he_normal'),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Dropout(0.3),

    Dense(1024, kernel_initializer='he_normal'),
    BatchNormalization(),    
    LeakyReLU(alpha=0.05),
    Dropout(0.3),

    Dense(10, activation='softmax', kernel_initializer='he_normal')
])
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
    rotation_range=15, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
)
datagen.fit(x_train)

# 모델 저장
today = date.today()
today_str = today.strftime("%Y-%m-%d-%H-%M")
today_weight_path = f"./weights/{today_str}/"

os.makedirs(today_weight_path, exist_ok=True)
file_path = today_weight_path+"epochs:{epoch:02d}-val_acc:{val_accuracy:.3f}-val_loss{val_loss:.3f}.hdf5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto'
)

# tensorboard -> $ tensorboard --logdir=./result_log/
log_dir = './result_log'
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1,
                                                      profile_batch=1
                                                      )
# learning_rate (Cosine Learning Rate Decay)
initial_learning_rate = 0.001
decay_steps = 55800
cosine_decay = tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(cosine_decay), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(datagen.flow(x_train, y_train, batch_size=512),
          epochs=300,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, model_checkpoint])
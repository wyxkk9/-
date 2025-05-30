import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Add, Multiply, GlobalAveragePooling2D, Reshape,
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ====================== 新增核心模块 ======================
def SEBlock(input_tensor, reduction_ratio=16):
    """ Squeeze-and-Excitation注意力模块 """
    channels = input_tensor.shape[-1]

    # Squeeze (全局平均池化)
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, channels))(x)

    # Excitation (两层全连接)
    x = Dense(channels // reduction_ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)

    # 特征重标定
    return Multiply()([input_tensor, x])


def ResBlock(x, filters, kernel_size=3, stride=1):
    """ 残差连接块 """
    shortcut = x

    # 主分支
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # 捷径分支适配维度
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # 合并分支
    x = Add()([x, shortcut])
    return Activation('relu')(x)


def build_improved_model(input_shape=(48, 48, 1)):
    """ 改进后的模型架构 """
    inputs = Input(shape=input_shape)

    # 初始卷积层
    x = Conv2D(32, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残差块1 + SE注意力
    x = ResBlock(x, 64)
    x = SEBlock(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)

    # 残差块2 + SE注意力
    x = ResBlock(x, 128)
    x = SEBlock(x)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.25)(x)

    # 残差块3
    x = ResBlock(x, 256)
    x = MaxPooling2D(2)(x)
    x = Dropout(0.5)(x)

    # 输出层
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(7, activation='softmax')(x)

    return Model(inputs, outputs)


# ====================== 数据准备 ======================
train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# ====================== 模型构建 ======================
emotion_model = build_improved_model(input_shape=(48, 48, 1))

cv2.ocl.setUseOpenCL(False)

# 表情标签
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# ====================== 模型训练 ======================
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy'])

# 添加训练回调
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

history = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=100,  # 增加训练轮次
    validation_data=validation_generator,
    validation_steps=7178 // 64,
    callbacks=callbacks
)

# 保存模型
emotion_model.save('improved_emotion_model.keras')


# ====================== 实时检测部分 ======================
def realtime_detection(model_path='improved_emotion_model.keras'):
    """ 修改后的实时检测函数 """
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]

            # 预处理
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)

            # 预测
            pred = model.predict(cropped_img)
            maxindex = int(np.argmax(pred))
            confidence = np.max(pred)  # 获取置信度

            # 显示结果（添加置信度）
            label = f"{emotion_dict[maxindex]} ({confidence:.2f})"
            cv2.putText(frame, label, (x + 20, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Improved Emotion Detection',
                   cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    realtime_detection()
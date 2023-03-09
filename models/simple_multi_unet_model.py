from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    concatenate,
    Conv2DTranspose,
    Dropout,
)
from keras import backend as K


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0
    )


################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1, dropout=0.1):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # Contraction path

    # e.g., (width, height) = (128, 128)
    # 128x128x1(input) -> 128x128x16(c1)
    c1 = Conv2D(16, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(inputs)
    c1 = Dropout(dropout)(c1)  # Original 0.1
    c1 = Conv2D(16, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(c1)
    # 128x128x16(c1) -> 64x64x16(p1)
    p1 = MaxPooling2D((2, 2))(c1)
    # 64x64x16(p1) -> 64x64x32(c2)
    c2 = Conv2D(32, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(dropout)(c2)  # Original 0.1
    c2 = Conv2D(32, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(c2)
    # 64x64x32(c2) -> 32x32x32(p2)
    p2 = MaxPooling2D((2, 2))(c2)
    # 32x32x32(p2) -> 32x32x64(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(dropout)(c3)
    c3 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    # 32x32x64(c3) -> 16x16x64(p3)
    p3 = MaxPooling2D((2, 2))(c3)
    # 32x32x64(p3) -> 16x16x128(c4)
    c4 = Conv2D(128, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(dropout)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu",
                kernel_initializer="he_normal", padding="same")(c4)
    # 16x16x128(c4) -> 8x8x128(p4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    # 8x8x128(p4) -> 8x8x256(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv2D(
        256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c5)

    # Expansive path
    # 8x8x256(c5) -> 16x16x128(u6) // u = up-sampling
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    # 16x16x128(u6), 16x16x128(c4) -> u6 = 32x32x128(u6, c4)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv2D(
        128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c6)
    # 32x32x128(c6) -> 32x32x64(u7)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    # 32x32x64(u7), 32x32x64(c3) -> u7 = 64x64x64(u7, c3)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv2D(
        64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c7)
    # 64x64x64(c7) -> 64x64x32(u8)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    # 64x64x32(c2), 64x64x32(u8) -> u8 = 128x128x32(u8, c2)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(
        32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c8)
    # 128x128x32(c8) -> 128x128x16(u9)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    # 128x128x16(c1), 128x128x16(u9) -> u9 = 128x128x32(u9, c1)//axis=3: concatenate along the channel axis
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(u9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(
        16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
    )(c9)
    # 128x128x32(c9) -> 128x128x2(outputs) -> 2 layers: background, foreground
    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    # NOTE: Compile the model in the main program to make it easy to test with various loss functions
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    return model

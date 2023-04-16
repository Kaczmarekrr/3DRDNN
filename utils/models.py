from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    BatchNormalization,
    Add,
)


def build_model_2DUNET(input_layer, start_neurons):
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        input_layer
    )
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        uconv4
    )
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        uconv4
    )

    # 16 -> 32
    deconv3 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        uconv3
    )
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        uconv3
    )

    # 32 -> 64
    deconv2 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        uconv2
    )
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        uconv2
    )

    # 64 -> 128
    deconv1 = Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        uconv1
    )
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        uconv1
    )

    # uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(2, (1, 1), padding="same", activation="softmax")(uconv1)

    return output_layer


def build_model_3DUNET(input_layer, start_neurons):
    # 128 -> 64
    conv1 = Conv3D(start_neurons * 1, (3, 3, 3), activation="relu", padding="same")(
        input_layer
    )

    conv1 = BatchNormalization()(conv1)

    conv1 = Conv3D(start_neurons * 2, (1, 3, 3), activation="relu", padding="same")(
        conv1
    )

    pool1 = MaxPooling3D((2, 2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32

    conv2 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(
        pool1
    )

    conv2 = BatchNormalization()(conv2)

    conv2 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(
        conv2
    )

    pool2 = MaxPooling3D((2, 2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(
        pool2
    )

    normalization3 = BatchNormalization()(conv3)

    conv3 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(
        normalization3
    )
    pool3 = MaxPooling3D((2, 2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Dropout(0.5)(pool3)

    # Middle
    convm = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(
        pool3
    )

    normalization4 = BatchNormalization()(convm)

    convm = Conv3D(start_neurons * 16, (3, 3, 3), activation="relu", padding="same")(
        normalization4
    )

    # 16 -> 32
    deconv3 = Conv3DTranspose(
        start_neurons * 16, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)

    uconv3 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(
        uconv3
    )

    normalization5 = BatchNormalization()(uconv3)

    uconv3 = Conv3D(start_neurons * 8, (3, 3, 3), activation="relu", padding="same")(
        normalization5
    )

    uconv2 = BatchNormalization()(uconv3)
    # 32 -> 64
    deconv2 = Conv3DTranspose(
        start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)

    uconv2 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(
        uconv2
    )

    normalization6 = BatchNormalization()(uconv2)

    uconv2 = Conv3D(start_neurons * 4, (3, 3, 3), activation="relu", padding="same")(
        normalization6
    )

    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    deconv1 = Conv3DTranspose(
        start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)

    uconv1 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(
        uconv1
    )

    normalization7 = BatchNormalization()(uconv1)

    uconv1 = Conv3D(start_neurons * 2, (3, 3, 3), activation="relu", padding="same")(
        normalization7
    )

    output_layer = Conv3D(2, (1, 1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


def build_model_3DUNET_CH(input_layer, start_neurons):
    # 128 -> 64
    conv11 = Conv3D(start_neurons * 1, (1, 3, 3), activation="relu", padding="same")(
        input_layer
    )
    conv12 = Conv3D(start_neurons * 1, (3, 1, 3), activation="relu", padding="same")(
        input_layer
    )
    conv13 = Conv3D(start_neurons * 1, (3, 3, 1), activation="relu", padding="same")(
        input_layer
    )
    conv1 = Add()([conv11, conv12, conv13])

    conv1 = BatchNormalization()(conv1)

    conv11 = Conv3D(start_neurons * 2, (1, 3, 3), activation="relu", padding="same")(
        conv1
    )
    conv12 = Conv3D(start_neurons * 2, (3, 1, 3), activation="relu", padding="same")(
        conv1
    )
    conv13 = Conv3D(start_neurons * 2, (3, 3, 1), activation="relu", padding="same")(
        conv1
    )
    conv1 = Add()([conv11, conv12, conv13])

    # conv1 = Convolution3DCH(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    # conv1 = Convolution3DCH(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)

    pool1 = MaxPooling3D((2, 2, 2))(conv1)
    pool1 = BatchNormalization()(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32

    conv21 = Conv3D(start_neurons * 2, (1, 3, 3), activation="relu", padding="same")(
        pool1
    )
    conv22 = Conv3D(start_neurons * 2, (3, 1, 3), activation="relu", padding="same")(
        pool1
    )
    conv23 = Conv3D(start_neurons * 2, (3, 3, 1), activation="relu", padding="same")(
        pool1
    )
    conv2 = Add()([conv21, conv22, conv23])

    conv2 = BatchNormalization()(conv2)

    conv21 = Conv3D(start_neurons * 4, (1, 3, 3), activation="relu", padding="same")(
        conv2
    )
    conv22 = Conv3D(start_neurons * 4, (3, 1, 3), activation="relu", padding="same")(
        conv2
    )
    conv23 = Conv3D(start_neurons * 4, (3, 3, 1), activation="relu", padding="same")(
        conv2
    )
    conv2 = Add()([conv21, conv22, conv23])

    pool2 = MaxPooling3D((2, 2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv31 = Conv3D(start_neurons * 4, (1, 3, 3), activation="relu", padding="same")(
        pool2
    )
    conv32 = Conv3D(start_neurons * 4, (3, 1, 3), activation="relu", padding="same")(
        pool2
    )
    conv33 = Conv3D(start_neurons * 4, (3, 3, 1), activation="relu", padding="same")(
        pool2
    )
    conv3 = Add()([conv31, conv32, conv33])

    normalization3 = BatchNormalization()(conv3)

    conv31 = Conv3D(start_neurons * 8, (1, 3, 3), activation="relu", padding="same")(
        normalization3
    )
    conv32 = Conv3D(start_neurons * 8, (3, 1, 3), activation="relu", padding="same")(
        normalization3
    )
    conv33 = Conv3D(start_neurons * 8, (3, 3, 1), activation="relu", padding="same")(
        normalization3
    )
    conv3 = Add()([conv31, conv32, conv33])

    pool3 = MaxPooling3D((2, 2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)
    pool3 = Dropout(0.5)(pool3)

    # Middle
    convm1 = Conv3D(start_neurons * 8, (1, 3, 3), activation="relu", padding="same")(
        pool3
    )
    convm2 = Conv3D(start_neurons * 8, (3, 1, 3), activation="relu", padding="same")(
        pool3
    )
    convm3 = Conv3D(start_neurons * 8, (3, 3, 1), activation="relu", padding="same")(
        pool3
    )
    convm = Add()([convm1, convm2, convm3])

    normalization4 = BatchNormalization()(convm)

    convm1 = Conv3D(start_neurons * 16, (1, 3, 3), activation="relu", padding="same")(
        normalization4
    )
    convm2 = Conv3D(start_neurons * 16, (3, 1, 3), activation="relu", padding="same")(
        normalization4
    )
    convm3 = Conv3D(start_neurons * 16, (3, 3, 1), activation="relu", padding="same")(
        normalization4
    )
    convm = Add()([convm1, convm2, convm3])

    # 16 -> 32
    deconv3 = Conv3DTranspose(
        start_neurons * 16, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)

    uconv31 = Conv3D(start_neurons * 8, (1, 3, 3), activation="relu", padding="same")(
        uconv3
    )
    uconv32 = Conv3D(start_neurons * 8, (3, 1, 3), activation="relu", padding="same")(
        uconv3
    )
    uconv33 = Conv3D(start_neurons * 8, (3, 3, 1), activation="relu", padding="same")(
        uconv3
    )
    uconv3 = Add()([uconv31, uconv32, uconv33])

    normalization5 = BatchNormalization()(uconv3)

    uconv31 = Conv3D(start_neurons * 8, (1, 3, 3), activation="relu", padding="same")(
        normalization5
    )
    uconv32 = Conv3D(start_neurons * 8, (3, 1, 3), activation="relu", padding="same")(
        normalization5
    )
    uconv33 = Conv3D(start_neurons * 8, (3, 3, 1), activation="relu", padding="same")(
        normalization5
    )
    uconv3 = Add()([uconv31, uconv32, uconv33])

    uconv2 = BatchNormalization()(uconv3)
    # 32 -> 64
    deconv2 = Conv3DTranspose(
        start_neurons * 8, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)

    uconv21 = Conv3D(start_neurons * 4, (1, 3, 3), activation="relu", padding="same")(
        uconv2
    )
    uconv22 = Conv3D(start_neurons * 4, (3, 1, 3), activation="relu", padding="same")(
        uconv2
    )
    uconv23 = Conv3D(start_neurons * 4, (3, 3, 1), activation="relu", padding="same")(
        uconv2
    )
    uconv2 = Add()([uconv21, uconv22, uconv23])

    normalization6 = BatchNormalization()(uconv2)

    uconv21 = Conv3D(start_neurons * 4, (1, 3, 3), activation="relu", padding="same")(
        normalization6
    )
    uconv22 = Conv3D(start_neurons * 4, (3, 1, 3), activation="relu", padding="same")(
        normalization6
    )
    uconv23 = Conv3D(start_neurons * 4, (3, 3, 1), activation="relu", padding="same")(
        normalization6
    )
    uconv2 = Add()([uconv21, uconv22, uconv23])

    uconv2 = BatchNormalization()(uconv2)

    # 64 -> 128
    deconv1 = Conv3DTranspose(
        start_neurons * 4, (3, 3, 3), strides=(2, 2, 2), padding="same"
    )(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)

    uconv11 = Conv3D(start_neurons * 2, (1, 3, 3), activation="relu", padding="same")(
        uconv1
    )
    uconv12 = Conv3D(start_neurons * 2, (3, 1, 3), activation="relu", padding="same")(
        uconv1
    )
    uconv13 = Conv3D(start_neurons * 2, (3, 3, 1), activation="relu", padding="same")(
        uconv1
    )
    uconv1 = Add()([uconv11, uconv12, uconv13])

    normalization7 = BatchNormalization()(uconv1)

    uconv11 = Conv3D(start_neurons * 2, (1, 3, 3), activation="relu", padding="same")(
        normalization7
    )
    uconv12 = Conv3D(start_neurons * 2, (3, 1, 3), activation="relu", padding="same")(
        normalization7
    )
    uconv13 = Conv3D(start_neurons * 2, (3, 3, 1), activation="relu", padding="same")(
        normalization7
    )
    uconv1 = Add()([uconv11, uconv12, uconv13])

    output_layer = Conv3D(2, (1, 1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


def model_call(model_name: str, pxz: int, px: int, features: int):
    if model_name == "2DUNET":
        input_layer = Input((px, px, 1))
        output_layer = build_model_2DUNET(input_layer, features)
        model = Model(input_layer, output_layer)
        return model

    elif model_name == "3DUNET":
        input_layer = Input((pxz, px, px, 1))
        output_layer = build_model_3DUNET(input_layer, features)
        model = Model(input_layer, output_layer)
        return model
    elif model_name == "3DUNET_CH":
        output_layer = build_model_3DUNET_CH(input_layer, features)
        model = Model(input_layer, output_layer)
        return model
    else:
        raise ValueError("wrong model_name -> works: ['2DUNET','3DUNET]")

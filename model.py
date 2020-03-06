from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from sru import SRU


def sim_unet(input_size=(256, 256, 1)):
    init_filter = 64
    depth = 4
    inputs = Input(shape=input_size)
    conv = inputs
    down_layer = []
    for i in range(depth):
        conv = rec_res_block(conv, init_filter * (2 ** i))
        down_layer.append(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = normal_conv(1024, (3, 3), conv)
    start_index = depth - 1
    for i in range(start_index, -1, -1):
        merge_roll = up_and_concate(init_filter * (2 ** i), conv, down_layer[i])
        conv = rec_res_block(merge_roll, init_filter * (2 ** i))
    conv = Conv2D(2, (3, 3), activation='relu', padding='same')(conv)
    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    model = Model(inputs=inputs, output=conv)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def seq_unet(pretrained_weights=None, input_size=(256, 256, 1)):
    init_filter = 64
    depth = 4
    inputs = Input(shape=input_size)
    conv = inputs
    down_layer = []
    for i in range(depth):
        conv = normal_conv(init_filter * (2 ** i), (3, 3), conv)
        # conv = rec_res_block(conv, init_filter*(2**i))
        down_layer.append(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

    # 到底了
    conv = normal_conv(1024, (3, 3), conv)
    for i in range(depth - 1, -1, -1):
        # merge_roll = up_and_concate(init_filter*(2**i), conv, down_layer[i])
        merge_roll = attention_up_and_concate(conv, down_layer[i])
        # conv = rec_res_block(merge_roll, init_filter*(2**i))
        conv = normal_conv(init_filter * (2 ** i), (3, 3), merge_roll)

    conv = Conv2D(2, (3, 3), activation='relu', padding='same')(conv)
    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    model = Model(inputs=inputs, output=conv)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    conv = Input(shape=input_size)
    # for i in range(depth):
    #     conv = normal_conv(init_filter*(2**i), (3, 3), conv)
    #     down_layer.append(conv)
    #     conv = MaxPooling2D(pool_size=(2, 2))(conv)

    conv1 = normal_conv(64, (3, 3), conv)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = normal_conv(128, (3, 3), pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = normal_conv(256, (3, 3), pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = normal_conv(512, (3, 3), pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 尝试
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 到底了
    #     conv5 = normal_conv(1024, (3, 3), pool4)
    conv5 = normal_conv(1024, (3, 3), pool4)
    # drop5 = Dropout(0.5)(conv5)
    drop5 = conv5  # 尝试
    # attention_up_and_concate(drop5, conv4)

    # 爬升
    # 1、加入注意力模块
    # 2、Recurrent Residual卷积
    # merge6 = up_and_concate(512, drop5, conv4)
    merge6 = attention_up_and_concate(drop5, conv4)
    conv6 = normal_conv(512, (3, 3), merge6)

    # merge7 = up_and_concate(256, conv6, conv3)
    merge7 = attention_up_and_concate(conv6, conv3)
    conv7 = normal_conv(256, (3, 3), merge7)

    # merge8 = up_and_concate(128, conv7, conv2)
    merge8 = attention_up_and_concate(conv7, conv2)
    conv8 = normal_conv(128, (3, 3), merge8)

    # merge9 = up_and_concate(64, conv8, conv1)
    merge9 = attention_up_and_concate(conv8, conv1)
    conv9 = normal_conv(64, (3, 3), merge9)

    conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=conv, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def normal_conv(input_layer, filters, conv):
    conv = Conv2D(input_layer, filters, activation='relu', padding='same')(conv)
    return Conv2D(input_layer, filters, activation='relu', padding='same')(conv)


def attention_block_2d(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1))(x)
    phi_g = Conv2D(inter_channel, (1, 1))(g)
    f = Conv2D(inter_channel, (2, 2), activation='relu', padding='same')(add([theta_x, phi_g]))
    rate = Conv2D(1, 1, activation='sigmoid', padding='same')(f)
    att_x = multiply([x, rate])
    return att_x


def attention_up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    up = Conv2D(in_channel // 2, (2, 2), activation='relu', padding='same')(up)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 2)
    return concatenate([up, layer], axis=3)


def rec_res_block(input_layer, out_n_filters, kernel_size=(3, 3),
                  padding='same'):
    input_n_filters = input_layer.get_shape().as_list()[3]
    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, (3, 3), padding=padding)(input_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, padding=padding, activation='relu')(
                    layer)
            layer1 = Conv2D(out_n_filters, kernel_size, padding=padding, activation='relu')(
                add([layer1, layer]))
        layer = layer1
    out_layer = add([layer, skip_layer])
    return out_layer


def normal_conv(input_layer, filters, conv):
    conv = Conv2D(input_layer, filters, activation='relu', padding='same')(conv)
    return Conv2D(input_layer, filters, activation='relu', padding='same')(conv)


def up_and_concate(input_layer, down_layer, layer):
    up = UpSampling2D(size=2)(down_layer)
    conv = Conv2D(input_layer, (2, 2), activation='relu', padding="same")(up)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([conv, layer])
    return concate


def recurrent_conv(input_layer, filters, conv):
    conv = Conv2D(input_layer, filters, activation='relu', padding='same')(conv)
    reshape = Lambda(lambda x: tf.reshape(tensor=x, shape=(-1, conv.shape[1], conv.shape[2]*conv.shape[3])))(conv)
    units = conv.shape[1]*conv.shape[2]*conv.shape[3]
    # gru = GRU(units=int(units))(reshape)
    print("test")
    gru = SRU(units=int(units), dropout=0.0, recurrent_dropout=0.0, unroll=True)
    gru = Dense(units)(gru)
    print("Hello")
    reshape = Lambda(lambda x: tf.reshape(tensor=x, shape=(-1, conv.shape[1], conv.shape[2], conv.shape[3])))(gru)
    return Conv2D(input_layer, filters, activation='relu', padding='same')(reshape)


# def test_net(input_size=(256, 256, 1)):
#     inputs = Input(shape=input_size)
#     conv = Conv2D(64, 3, activation='relu', padding='same')(inputs)
#     reshape = Lambda(reshape_tensor, arguments={'shape': (-1, conv.shape[1], conv.shape[2]*conv.shape[3])})(conv)
#     gru = GRU(units=int(conv.shape[1]*conv.shape[2]*conv.shape[3]))(reshape)
#     reshape = Lambda(reshape_tensor, arguments={'shape': (-1, conv.shape[1], conv.shape[2], conv.shape[3])})(gru)
#     conv = Conv2D(64, 3, activation='relu', padding='same')(reshape)
#     model = Model(inputs=inputs, output=conv)
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#     return model


def res_attention_Unet(input_size=(256, 256, 1)):
    init_filter = 64
    depth = 4
    inputs = Input(shape=input_size)
    conv = inputs
    down_layer = []
    for i in range(depth):
        conv = rec_res_block(conv, init_filter * (2 ** i))
        down_layer.append(conv)
        conv = MaxPooling2D(pool_size=(2, 2))(conv)
    conv = recurrent_conv(init_filter * (2 ** i), (3, 3), conv)
    # conv = normal_conv(init_filter * (2 ** i), (3, 3), conv)
    start_index = depth - 1
    for i in range(start_index, -1, -1):
        merge_roll = up_and_concate(init_filter * (2 ** i), conv, down_layer[i])
        conv = rec_res_block(merge_roll, init_filter * (2 ** i))
    conv = Conv2D(2, (3, 3), activation='relu', padding='same')(conv)
    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    model = Model(inputs=[inputs], output=[conv])
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


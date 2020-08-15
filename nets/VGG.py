import tensorflow as tf
import numpy as np
import tf_slim as slim
import cv2
import os

def vgg_arg_scope(weight_decay=0.0005):

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16_base(inputs,
           is_training=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           final_endpoint = None):
    """
    VGG16模型
    :param inputs:a tensor [batch_size, height, width, channels]
    :param num_classes:分类数
    :param is_training: 是否训练
    :param dropout_keep_prob: 训练时dropout保持激活的可能性
    :param spatial_squeeze:是否压缩输出的空间维度
    :param scope:变量的可选范围
    :param fc_conv_padding: 全连接层的填充类型 'SAME' or 'VALID'
    :param global_pool: a boolean flag .True: 则对分类模块的输入需用平均池化
    :return: net: VGG net
             end_points :a dict of tensors with intermediate activations.
    """
    end_points ={}
    with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            end_point = 'pool4'
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net,end_points

            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')

            end_point = 'conv5_2'
            end_points[end_point] = net
            if end_point == final_endpoint:
                return net,end_points

            net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # # Convert end_points_collection into a end_point dict.
            # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           reuse=False,
           global_pools = False,
           feature_maps = 'pool4',
           attention_maps = 'conv5_2',
           num_parts = 32):

    with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs],reuse = reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = vgg_16_base(inputs,
                                          final_endpoint='pool5'
                                          )
            with tf.compat.v1.variable_scope('bilinear_attention_pooling'):
                feature_maps = end_points[feature_maps]
                attention_maps = end_points[attention_maps]

                attention_maps = attention_maps[:, :, :, :num_parts]

                attention_image = tf.compat.v1.py_func(generate_attention_image, [inputs[0], attention_maps[0]], tf.uint8)
                tf.summary.image('attention_image', tf.expand_dims(attention_image, 0))
                tf.summary.image('input_image', inputs[0:1])
                if is_training:
                    tf.summary.image('attention_maps', tf.reduce_mean(attention_maps[0:1], axis=-1, keepdims=True))
                    tf.summary.image('feature_maps', tf.reduce_mean(feature_maps[0:1], axis=-1, keepdims=True))

                end_points['attention_maps'] = attention_maps
                end_points[ 'feature_maps' ] = feature_maps

                bap_features, end_points = bilinear_attention_pooling(feature_maps, attention_maps
                                                                      , end_points, 'embeddings')

                # # 使用卷积层代替全连接层
                # net = slim.conv2d(bap_features, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                    scope='dropout6')
                # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                if global_pools:
                    bap_features = tf.reduce_mean(bap_features, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = bap_features
                if num_classes:
                    bap_features = slim.dropout(bap_features, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    bap_features = slim.conv2d(bap_features, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        bap_features = tf.squeeze(bap_features, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = bap_features


            return bap_features, end_points


vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
    """
        VGG19模型
        :param inputs:a tensor [batch_size, height, width, channels]
        :param num_classes:分类数
        :param is_training: 是否训练
        :param dropout_keep_prob: 训练时dropout保持激活的可能性
        :param spatial_squeeze:是否压缩输出的空间维度
        :param scope:变量的可选范围
        :param fc_conv_padding: 全连接层的填充类型 'SAME' or 'VALID'
        :param global_pool: a boolean flag .True: 则对分类模块的输入需用平均池化
        :return: net: VGG net
                 end_points :a dict of tensors with intermediate activations.
        """
    with tf.compat.v1.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points

vgg_19.default_image_size = 224

def bilinear_attention_pooling(feature_maps, attention_maps, end_points, name):
    feature_shape = feature_maps.get_shape().as_list()
    attention_shape = attention_maps.get_shape().as_list()

    phi_I = tf.einsum('ijkm,ijkn->imn', attention_maps, feature_maps)
    phi_I = tf.divide(phi_I, tf.cast(attention_shape[1] * attention_shape[2],dtype=tf.float32))
    phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))

    raw_features = tf.nn.l2_normalize(phi_I, axis=[1, 2])
    raw_features = tf.reshape(raw_features, [-1, 1, 1, attention_shape[-1] * feature_shape[-1]])
    end_points[name] = raw_features

    pooling_features = raw_features * 100.0
    return pooling_features, end_points

def wsddn_pooling(feature_maps, attention_maps, keep_prob, end_points, name):
    feature_shape = feature_maps.get_shape().as_list()
    attention_shape = attention_maps.get_shape().as_list()

    phi_I = tf.einsum('ijkm,ijkn->imn', attention_maps, feature_maps)
    phi_I = tf.divide(phi_I, tf.cast(attention_shape[1] * attention_shape[2],dtype=tf.float32))
    phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))

    phi_attention = phi_I / tf.reduce_sum(phi_I + 1e-12, axis=1, keepdims=True)
    phi_feature = phi_I / tf.reduce_sum(phi_I + 1e-12, axis=2, keepdims=True)
    phi_I = phi_attention * phi_feature

    pooling_features = tf.reduce_sum(phi_I, axis=1)
    pooling_features = tf.nn.l2_normalize(pooling_features, axis=-1)
    end_points[name] = tf.reshape(pooling_features, [-1, 1, 1, feature_shape[-1]])

    pooling_features = tf.reshape(pooling_features * 100.0, [-1, 1, 1, feature_shape[-1]])

    return pooling_features, end_points

def aspp_residual(attention_maps):
    scale1 = attention_maps
    scale2 = slim.conv2d(attention_maps, 192, [3, 3], rate=6)
    scale3 = slim.conv2d(attention_maps, 192, [3, 3], rate=12)
    scale4 = slim.conv2d(attention_maps, 192, [3, 3], rate=18)
    attention_maps = scale1 + scale2 + scale3 + scale4
    return attention_maps

def generate_attention_image(image, attention_map):
    h, w, _ = image.shape
    mask = np.mean(attention_map, axis=-1, keepdims=True)
    mask = (mask / np.max(mask) * 255.0).astype(np.uint8)
    mask = cv2.resize(mask, (w, h))

    image = (image / 2.0 + 0.5) * 255.0
    image = image.astype(np.uint8)

    color_map = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
    attention_image = cv2.addWeighted(image, 0.5, color_map.astype(np.uint8), 0.5, 0)
    attention_image = cv2.cvtColor(attention_image, cv2.COLOR_BGR2RGB)
    return attention_image

import tensorflow.keras as keras
from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

def Vgg16_base(inputs,
               final_point = 'block5_conv2',
               **kwargs):

    end_points = {}

    # Block 1
    x = keras.layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1')(inputs)
    x = keras.layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    end_point = 'block4_pool'
    end_points[end_point] = x
    if end_point == final_point:
        return x, end_points

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block5_conv2')(x)
    end_point = 'block5_conv2'
    end_points[end_point] = x
    if end_point == final_point:
        return x, end_points

    return x, end_points


def Vgg16(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          num_classes = 1000,
          final_point = 'block5_conv2',
          feature_maps='pool4',
          attention_maps='pool5',
          num_parts = 32,
          is_training = True,
          spatial_squeeze=True,
          global_pools=False,
          **kwargs):
    '''
    :param include_top:
    :param weights:
    :param input_tensor:
    :param input_shape:
    :param num_classes:
    :param final_point:
    :param feature_maps:
    :param attention_maps:
    :param num_parts:
    :param is_training:
    :param spatial_squeeze:
    :param global_pools:
    :param kwargs:
    :return: model : a keras model
            bap_features: 同 vgg_16
            end_points:  同 vgg_16
    '''
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and num_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    net, end_points = Vgg16_base(img_input,final_point='block5_conv2')

    x = keras.layers.Conv2D(512, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block5_conv3')(net)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    feature_maps = end_points[feature_maps]
    attention_maps = end_points[attention_maps]

    attention_maps = attention_maps[:, :, :, :num_parts]

    attention_image = tf.compat.v1.py_func(generate_attention_image, [inputs[0], attention_maps[0]], tf.uint8)
    tf.summary.image('attention_image', tf.expand_dims(attention_image, 0))
    tf.summary.image('input_image', inputs[0:1])
    if is_training:
        tf.summary.image('attention_maps', tf.reduce_mean(attention_maps[0:1], axis=-1, keepdims=True))
        tf.summary.image('feature_maps', tf.reduce_mean(feature_maps[0:1], axis=-1, keepdims=True))

    end_points['attention_maps'] = attention_maps
    end_points['feature_maps'] = feature_maps

    bap_features, end_points = bilinear_attention_pooling(feature_maps, attention_maps
                                                          , end_points, 'embeddings')
    if global_pools:
        bap_features = tf.reduce_mean(bap_features, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = bap_features
    if num_classes:
        x = layers.Flatten(name='flatten')(bap_features)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        bap_features = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        if spatial_squeeze:
            bap_features = tf.squeeze(bap_features, [1, 2], name='fc8/squeezed')
        end_points['vgg_16' + '/fc8'] = bap_features

    # Create model.
    model = models.Model(inputs, bap_features, name='vgg16')

    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = keras_utils.get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model, bap_features,end_points
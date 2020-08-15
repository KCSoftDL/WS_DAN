import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf_v1
import tensorflow as tf
import tf_slim as slim
import random
import shutil
import math
import matplotlib.pyplot as plt
import scipy.io as sio

import convert_data
from nets import net_select
import preprocessing_select
import vgg_preprocessing

batch_size = 16
num_preprocessing_threads = 4
moving_average_decay = None
max_num_batches = None

def add_eval_summary(logits, labels, scope=''):
    predictions = tf_v1.argmax(logits, 1)
    labels = tf.squeeze(labels)
    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval%s/%s' % (scope, name)
        op = tf_v1.summary.scalar(summary_name, value, collections=[])
        op = tf.print(op, [value], summary_name)
        tf_v1.add_to_collection(tf_v1.GraphKeys.SUMMARIES, op)
    return names_to_updates


def draw_keypints(image, keypoints, exist):
    for i in range(exist.size):
        if exist[i] == 1:
            cv2.circle(image, center=(int(keypoints[2 * i]), int(keypoints[2 * i + 1])), radius=5, color=(255, 0, 0),
                       thickness=-1)

    return image


def visualization(images, feature_maps, logits):
    index_dir = str(random.randint(0, 100))
    visual_dir = os.path.join('./Stanford-Cars/visualization', index_dir)

    if os.path.exists(visual_dir):
        shutil.rmtree(visual_dir)
    os.makedirs(visual_dir)

    img = ((images[0] + 1) * 127).astype(np.uint8)
    # img = (images[0] + 128).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(visual_dir, 'image.jpg'), img)

    feature_map = feature_maps[0]
    mean_feature = np.mean(feature_map, axis=-1, keepdims=True)
    mean_feature = (mean_feature / np.max(mean_feature) * 255).astype(np.uint8)
    mean_feature = cv2.resize(mean_feature, (100, 100))
    cv2.imwrite(os.path.join(visual_dir, 'mean_feature.jpg'), mean_feature)

    max_feature = np.max(feature_map, axis=-1, keepdims=True)
    max_feature = (max_feature / np.max(max_feature) * 255).astype(np.uint8)
    max_feature = cv2.resize(max_feature, (100, 100))
    cv2.imwrite(os.path.join(visual_dir, 'max_feature.jpg'), max_feature)

    feature_map = (feature_map / np.max(feature_map) * 255).astype(np.uint8)
    for index in range(feature_maps.shape[-1]):
        feature = np.expand_dims(feature_map[:, :, index], axis=2)
        feature = cv2.resize(feature, (100, 100))
        cv2.imwrite(os.path.join(visual_dir, '%s.jpg' % index), feature)

    return logits


def predict_results(images, feature_maps, logits, labels):
    for i in range(images.shape[0]):
        image = images[i]
        label = labels[i]
        logit = logits[i]

        index_dir = str(np.argmax(logit))
        visual_dir = os.path.join('./Stanford-Cars/predict_results', index_dir)
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

        img = ((image + 1) * 127).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_name = str(label) + '_' + str(random.randint(1, 10000)) + '.jpg'

        feature_map = feature_maps[i]
        mean_feature = np.mean(feature_map, axis=-1, keepdims=True)
        mean_feature = (mean_feature / np.max(mean_feature, keepdims=True) * 255).astype(np.uint8)
        mean_feature = cv2.resize(mean_feature, (image.shape[0], image.shape[1]))
        mean_feature = np.reshape(mean_feature, [image.shape[0], image.shape[1], 1])

        mean_feature = np.tile(mean_feature, [1, 1, 3])

        showImg = np.concatenate([img, mean_feature], axis=1)
        cv2.imwrite(os.path.join(visual_dir, image_name), showImg)

    return logits


def mask2bbox(attention_maps):
    height = attention_maps.shape[1]
    width = attention_maps.shape[2]
    bboxes = []
    for i in range(attention_maps.shape[0]):
        mask = attention_maps[i]
        max_activate = mask.max()
        min_activate = 0.1 * max_activate
        mask = (mask >= min_activate)
        itemindex = np.where(mask == True)

        ymin = itemindex[0].min() / height - 0.05
        ymax = itemindex[0].max() / height + 0.05
        xmin = itemindex[1].min() / width - 0.05
        xmax = itemindex[1].max() / width + 0.05

        bbox = np.asarray([ymin, xmin, ymax, xmax], dtype=np.float32)
        bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def main(model_root,datasets_dir,model_name,test_image_name):
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        test_image = os.path.join(datasets_dir,test_image_name)


        dataset = convert_data.get_datasets('train',dataset_dir=datasets_dir)

        network_fn = net_select.get_network_fn(
            model_name,
            num_classes=dataset.num_classes,
            is_training=False)

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=20 * batch_size,
            common_queue_min=10 * batch_size)
        [image, label] = provider.get(['image', 'label'])

        image_preprocessing_fn = preprocessing_select.get_preprocessing(
            model_name,
            is_training=False)

        eval_image_size = network_fn.default_image_size
        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf_v1.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocessing_threads,
            capacity=5 * batch_size)

        checkpoint_path = os.path.join(model_root, model_name)
        if tf.io.gfile.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path

        logits_1, end_points_1 = network_fn(images)
        attention_maps = tf.reduce_mean(end_points_1['attention_maps'], axis=-1, keepdims=True)
        attention_maps = tf.image.resize(attention_maps, [eval_image_size, eval_image_size],
                                         method=tf.image.ResizeMethod.BILINEAR)
        bboxes = tf_v1.py_func(mask2bbox, [attention_maps], [tf.float32])
        bboxes = tf.reshape(bboxes, [batch_size, 4])
        box_ind = tf.range(batch_size, dtype=tf.int32)

        images = tf.image.crop_and_resize(images, bboxes, box_ind, crop_size=[eval_image_size, eval_image_size])
        logits_2, end_points_2 = network_fn(images, reuse=True)

        logits = tf_v1.log(tf.nn.softmax(logits_1) * 0.5 + tf.nn.softmax(logits_2) * 0.5)


        """
        tf_v1.enable_eager_execution()

        #测试单张图片
        image_data = tf.io.read_file(test_image)
        image_data = tf.image.decode_jpeg(image_data,channels= 3)

        # plt.figure(1)
        # plt.imshow(image_data)

        image_data = image_preprocessing_fn(image_data, eval_image_size, eval_image_size)
        image_data = tf.expand_dims(image_data, 0)

        logits_3,end_points_3 = network_fn(image_data,reuse =True)
        attention_map = tf.reduce_mean(end_points_3['attention_maps'], axis=-1, keepdims=True)
        attention_map = tf.image.resize(attention_map, [eval_image_size, eval_image_size],
                                         method=tf.image.ResizeMethod.BILINEAR)
        bboxes = tf_v1.py_func(mask2bbox, [attention_map], [tf.float32])
        bboxes = tf.reshape(bboxes, [batch_size, 4])
        box_ind = tf.range(batch_size, dtype=tf.int32)

        image_data = tf.image.crop_and_resize(images, bboxes, box_ind, crop_size=[eval_image_size, eval_image_size])

        logits_4, end_points_4 = network_fn(image_data, reuse=True)
        logits_0 = tf_v1.log(tf.nn.softmax(logits_3) * 0.5 + tf.nn.softmax(logits_4) * 0.5)
        probabilities = logits_0[0,0:]

        print(probabilities)
        # sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),key= lambda x:x[1])]
        sorted_inds = (np.argsort(probabilities.numpy())[::-1])

        train_info = sio.loadmat(os.path.join(datasets_dir, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
        names = train_info['class']
        print(names)
        for i in range(5):
            index = sorted_inds[i]
            #  打印top5的预测类别和相应的概率值。
            print('Probability %0.2f => [%s]' % (probabilities[index],names[index+1][0][0]))
        """
        if moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()


        logits_to_updates = add_eval_summary(logits, labels, scope='/bilinear')
        logits_1_to_updates = add_eval_summary(logits_1, labels, scope='/logits_1')
        logits_2_to_updates = add_eval_summary(logits_2, labels, scope='/logits_2')

        if max_num_batches:
            num_batches = max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(batch_size))


        config = tf_v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        tf.compat.v1.disable_eager_execution()

        while True:
            if tf.io.gfile.isdir(checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            else:
                checkpoint_path = checkpoint_path

            print('Evaluating %s' % checkpoint_path)
            eval_op = []
            # eval_op = list(logits_to_updates.values())
            eval_op.append(list(logits_to_updates.values()))
            eval_op.append(list(logits_1_to_updates.values()))
            eval_op.append(list(logits_2_to_updates.values()))
            # tf.convert_to_tensor(eval_op)
            # tf.cast(eval_op,dtype=tf.string)
            # print(eval_op)

            test_dir = checkpoint_path
            slim.evaluation.evaluate_once(
                master=' ',
                checkpoint_path=checkpoint_path,
                logdir=test_dir,
                num_evals=num_batches,
                eval_op=eval_op,
                variables_to_restore=variables_to_restore,
                final_op=None,
                session_config=config)


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = vgg_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf_v1.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels

def predict(model_root,datasets_dir,model_name,test_image_name):
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        test_image = os.path.join(datasets_dir,test_image_name)


        # dataset = convert_data.get_datasets('test',dataset_dir=datasets_dir)

        network_fn = net_select.get_network_fn(
            model_name,
            num_classes=20,
            is_training=False)
        batch_size = 1
        eval_image_size = network_fn.default_image_size

        # images, images_raw, labels = load_batch(datasets_dir,
        #                                         height=eval_image_size,
        #                                         width=eval_image_size)

        image_preprocessing_fn = preprocessing_select.get_preprocessing(
            model_name,
            is_training=False)

        image_data = tf.io.read_file(test_image)
        image_data = tf.image.decode_jpeg(image_data,channels= 3)
        image_data = image_preprocessing_fn(image_data, eval_image_size, eval_image_size)
        image_data = tf.expand_dims(image_data, 0)

        logits_1, end_points_1 = network_fn(image_data)
        attention_maps = tf.reduce_mean(end_points_1['attention_maps'], axis=-1, keepdims=True)
        attention_maps = tf.image.resize(attention_maps, [eval_image_size, eval_image_size],
                                         method=tf.image.ResizeMethod.BILINEAR)
        bboxes = tf_v1.py_func(mask2bbox, [attention_maps], [tf.float32])
        bboxes = tf.reshape(bboxes, [batch_size, 4])
        # print(bboxes)
        box_ind = tf.range(batch_size, dtype=tf.int32)

        images = tf.image.crop_and_resize(image_data, bboxes, box_ind, crop_size=[eval_image_size, eval_image_size])
        logits_2, end_points_2 = network_fn(images, reuse=True)

        logits = tf.math.log(tf.nn.softmax(logits_1) * 0.5 + tf.nn.softmax(logits_2) * 0.5)

        checkpoint_path = os.path.join(model_root, model_name)

        if tf.io.gfile.isdir(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path

        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

        # with tf_v1.Session() as sess:
        #     with slim.queues.QueueRunners(sess):
        #         sess.run(tf_v1.initialize_local_variables())
        #         init_fn(sess)
        #         np_probabilities, np_images_raw, np_labels = sess.run([logits, images_raw, labels])
        #
        #         for i in range(batch_size):
        #             image = np_images_raw[i, :, :, :]
        #             true_label = np_labels[i]
        #             predicted_label = np.argmax(np_probabilities[i, :])
        #             print('true is {}, predict is {}'.format(true_label, predicted_label))

        with tf_v1.Session() as sess:
            with slim.queues.QueueRunners(sess):
                sess.run(tf_v1.initialize_local_variables())
                init_fn(sess)
                np_images, np_probabilities = sess.run([image_data, logits])
                predicted_label = np.argmax(np_probabilities[0, :])
                print(predicted_label)

'''
        # 因缺失label到class的对应关系，因此该部分暂无法实现。
        labels_to_names = None
        if convert_data.has_labels(datasets_dir):
            labels_to_names = convert_data.read_label_file(datasets_dir)

        class_name = labels_to_names[predicted_label]
        print(class_name)'''






if __name__ == '__main__':
    model_root = "./models"
    datasets_dir = "./datasets"
    test_image = "cars_test/00001.jpg"
    model_name = "vgg_16"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    predict(model_root= model_root,datasets_dir= datasets_dir,model_name=model_name,test_image_name= test_image)
    # main(model_root= model_root,datasets_dir= datasets_dir,model_name=model_name,test_image_name= test_image)
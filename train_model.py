import os
import tensorflow as tf
import numpy as np
import tf_slim as slim
import random

import model_deploy
import convert_data
from nets import net_select
import preprocessing_select


batch_size = 12
train_image_size = 448
num_clones = 1
num_ps_tasks = 8
num_readers = 8
max_number_of_steps= 80000
worker_replicas = 1
num_preprocessing_threads = 8
task = 0

replicas_to_aggregate = 1
num_epochs_per_decay = 2

learning_rate = 0.001
end_learning_rate = 0.0001
learning_rate_decay_type = "fixed"  # "fixed", "exponential",or "polynomial"
learning_rate_decay_factor = 0.9
label_smoothing = 0.0

moving_average_decay = None
weight_decay= 1e-5

log_every_n_steps = 10
save_summaries_secs = 60*2
save_interval_secs = 60*4
checkpoint_path = "D:/Programming/WS_DAN/models"
checkpoint_exclude_scopes = "vgg_16/bilinear_attention_pooling"
trainable_scopes = None
feature_maps="Mixed_6e"
attention_maps="Mixed_7a_b0"

def configure_learning_rate(num_samples_per_epoch, global_step):
    """
    配置学习率
    """
    decay_steps = int(num_samples_per_epoch / (batch_size * num_clones) *
                      num_epochs_per_decay)

    if learning_rate_decay_type == 'exponential':
        return tf.compat.v1.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate') # tf2.0中，该方法重写在keras中
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(learning_rate, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        return tf.compat.v1.train.polynomial_decay(learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         learning_rate_decay_type)

def configure_optimizer(learning_rate):
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate,
        beta1=0.9,
        beta2=0.99)
    return optimizer

def _get_init_fn(checkpoint_path,train_dir):
    if checkpoint_path is None:
        return None
    if tf.train.latest_checkpoint(train_dir):
        tf.compat.v1.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                break
        else:
            variables_to_restore.append(var)

    if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    tf.compat.v1.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)


def get_variables_to_train(trainable_scopes):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if trainable_scopes is None:
        return tf.compat.v1.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main(model_root,datasets_dir):
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # 训练相关参数设置
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=num_clones,
            clone_on_cpu=False,
            replica_id=task,
            num_replicas=worker_replicas,
            num_ps_tasks=num_ps_tasks)

        global_step = slim.create_global_step()

        model_name = 'vgg_16'
        train_dir =os.path.join(model_root, model_name)
        dataset = convert_data.get_datasets('train',dataset_dir=datasets_dir)

        network_fn = net_select.get_network_fn(
            model_name,
            num_classes=dataset.num_classes ,
            weight_decay=weight_decay,
            is_training=True)

        image_preprocessing_fn = preprocessing_select.get_preprocessing(
            model_name,
            is_training=True)

        print( "the data_sources:",dataset.data_sources)

        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size)
            [image, label] = provider.get(['image', 'label'])


            train_image_size = network_fn.default_image_size

            image = image_preprocessing_fn(image, train_image_size, train_image_size)

            images, labels = tf.compat.v1.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocessing_threads,
                capacity=5 * batch_size)
            labels = slim.one_hot_encoding(
                labels, dataset.num_classes )
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2 * deploy_config.num_clones)

        def calculate_pooling_center_loss(features, label, alfa, nrof_classes, weights, name):
            features = tf.reshape(features, [features.shape[0], -1])
            label = tf.argmax(label, 1)

            nrof_features = features.get_shape()[1]
            centers = tf.compat.v1.get_variable(name, [nrof_classes, nrof_features], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0), trainable=False)
            label = tf.reshape(label, [-1])
            centers_batch = tf.gather(centers, label)
            centers_batch = tf.nn.l2_normalize(centers_batch, axis=-1)

            diff = (1 - alfa) * (centers_batch - features)
            centers = tf.compat.v1.scatter_sub(centers, label, diff)

            with tf.control_dependencies([centers]):
                distance = tf.square(features - centers_batch)
                distance = tf.reduce_sum(distance, axis=-1)
                center_loss = tf.reduce_mean(distance)

            center_loss = tf.identity(center_loss * weights, name=name + '_loss')
            return center_loss

        def attention_crop(attention_maps):
            '''
            利用attention map 做数据增强,这里是论文中的Crop Mask
            :param attention_maps: Feature maps降维得到的
            :return:
            '''
            batch_size, height, width, num_parts = attention_maps.shape
            bboxes = []
            for i in range(batch_size):
                attention_map = attention_maps[i]
                part_weights = attention_map.mean(axis=0).mean(axis=0)
                part_weights = np.sqrt(part_weights)
                part_weights = part_weights / np.sum(part_weights)
                selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]

                mask = attention_map[:, :, selected_index]

                threshold = random.uniform(0.4, 0.6)
                itemindex = np.where(mask >= mask.max() * threshold)

                ymin = itemindex[0].min() / height - 0.1
                ymax = itemindex[0].max() / height + 0.1
                xmin = itemindex[1].min() / width - 0.1
                xmax = itemindex[1].max() / width + 0.1

                bbox = np.asarray([ymin, xmin, ymax, xmax], dtype=np.float32)
                bboxes.append(bbox)
            bboxes = np.asarray(bboxes, np.float32)
            return bboxes

        def attention_drop(attention_maps):
            '''
            这里是attention drop部分，目的是为了让模型可以注意到物体的其他部位（因不同attention map可能聚焦了同一部位）
            :param attention_maps:
            :return:
            '''
            batch_size, height, width, num_parts = attention_maps.shape
            masks = []
            for i in range(batch_size):
                attention_map = attention_maps[i]
                part_weights = attention_map.mean(axis=0).mean(axis=0)
                part_weights = np.sqrt(part_weights)
                if (np.sum(part_weights) != 0):
                    part_weights = part_weights / np.sum(part_weights)
                selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]
                mask = attention_map[:, :, selected_index:selected_index + 1]

                # soft mask
                threshold = random.uniform(0.2, 0.5)
                mask = (mask < threshold * mask.max()).astype(np.float32)
                masks.append(mask)
            masks = np.asarray(masks, dtype=np.float32)
            return masks

        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels = batch_queue.dequeue()
            logits_1, end_points_1 = network_fn(images)

            attention_maps = end_points_1['attention_maps']
            attention_maps = tf.image.resize(attention_maps, [train_image_size, train_image_size],
                                             method=tf.image.ResizeMethod.BILINEAR)

            # attention crop
            bboxes = tf.compat.v1.py_func(attention_crop, [attention_maps], [tf.float32])
            bboxes = tf.reshape(bboxes, [batch_size, 4])
            box_ind = tf.range(batch_size, dtype=tf.int32)
            images_crop = tf.image.crop_and_resize(images, bboxes, box_ind,
                                                   crop_size=[train_image_size, train_image_size])

            # attention drop
            masks = tf.compat.v1.py_func(attention_drop, [attention_maps], [tf.float32])
            masks = tf.reshape(masks, [batch_size, train_image_size, train_image_size, 1])
            images_drop = images * masks

            logits_2, end_points_2 = network_fn(images_crop, reuse=True)
            logits_3, end_points_3 = network_fn(images_drop, reuse=True)


            slim.losses.softmax_cross_entropy(logits_1, labels, weights=1 / 3.0, scope='cross_entropy_1')
            slim.losses.softmax_cross_entropy(logits_2, labels, weights=1 / 3.0, scope='cross_entropy_2')
            slim.losses.softmax_cross_entropy(logits_3, labels, weights=1 / 3.0, scope='cross_entropy_3')

            embeddings = end_points_1['embeddings']
            center_loss = calculate_pooling_center_loss(features=embeddings, label=labels,
                                                        alfa=0.95, nrof_classes=dataset.num_classes,
                                                        weights=1.0, name='center_loss')
            slim.losses.add_loss(center_loss)

            return end_points_1

        # Gather initial summaries.
        summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                                 first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOSSES,
                                                first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
            moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = configure_learning_rate(dataset.num_samples, global_step)
            optimizer = configure_optimizer(learning_rate)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = get_variables_to_train(trainable_scopes)

        #  and returns a train_tensor and summary_op
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)
        # Add total_loss to summary.
        summaries.add(tf.summary.scalar('total_loss', total_loss))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES,
                                               first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.compat.v1.summary.merge_all()

        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"

        save_model_path = os.path.join(checkpoint_path,model_name,"%s.ckpt"%model_name)
        print(save_model_path)

        # saver = tf.compat.v1.train.import_meta_graph('%s.meta'%save_model_path, clear_devices=True)
        tf.compat.v1.disable_eager_execution()
        # train the model
        slim.learning.train(
            train_op=train_tensor,
            logdir=train_dir,
            is_chief= (task == 0),
            init_fn= _get_init_fn(save_model_path,train_dir= train_dir),
            summary_op=summary_op,
            number_of_steps=max_number_of_steps,
            log_every_n_steps=log_every_n_steps,
            save_summaries_secs=save_summaries_secs,
            save_interval_secs=save_interval_secs,
            # sync_optimizer=None,
            session_config=config
            )

if __name__ == '__main__':
    model_root = "D:/Programming/WS_DAN/models"
    datasets_dir = "D:/Programming/WS_DAN/datasets"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    main(model_root= model_root,datasets_dir= datasets_dir)
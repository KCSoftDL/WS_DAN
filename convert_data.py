import tensorflow as tf
import random
import os
import sys
import math
import numpy as np
import tf_slim as slim
import scipy.io as sio

# Seed for repeatability.
_RANDOM_SEED = 0
# The number of shards per dataset split.
_NUM_SHARDS = 5

LABELS_FILENAME = 'labels.txt'

_FILE_PATTERN = 'food_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 145069, 'test': 20253}

_NUM_CLASSES = 208

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(image_data, image_format, height, width, label):

  return tf.train.Example(features=tf.train.Features(feature=
  {
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/class/label': int64_feature(label),
  }
  ))

def example_to_tfexample(image_data, image_format, height, width, label):

  return tf.train.Example(features=tf.train.Features(feature=
  {
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/class/label': int64_feature(label)
  }
  ))

def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):

  labels_filename = os.path.join(dataset_dir, filename)
  with tf.io.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):

  labels_filename = os.path.join(dataset_dir, filename)
  with tf.io.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

class ImageReader(object):

    def __init__(self):
        self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def get_filenames_and_classes(dataset_dir):

    image_root = os.path.join(dataset_dir, 'images')
    directories = []
    class_names = []
    for filename in os.listdir(image_root):
        path = os.path.join(image_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'Food_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    if not os.path.exists(os.path.join(dataset_dir, 'tfrecords')):
        os.makedirs(os.path.join(dataset_dir, 'tfrecords'))
    return os.path.join(dataset_dir, 'tfrecords', output_filename)

def has_labels(dataset_dir, filename=LABELS_FILENAME):
  return tf.io.gfile.exists(os.path.join(dataset_dir, filename))

def get_datasets(split_name, dataset_dir, file_pattern=None, reader=None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    dataset_dir = os.path.join(dataset_dir, 'tfrecords')
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # print(file_pattern)
    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        #reader = tf.data.TFRecordDataset
        reader = tf.compat.v1.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.io.FixedLenFeature(
        [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if has_labels(dataset_dir):
        labels_to_names = read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)


def convert_dataset(split_name, dataset, dataset_dir):

    assert split_name in ['train', 'test']

    num_per_shard = int(math.ceil(len(dataset) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0

        with tf.compat.v1.Session(config=config) as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(dataset))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting %s image %d/%d shard %d' %
                                         (split_name,i + 1, len(dataset), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.io.gfile.GFile(dataset[i]['filename'], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        label = dataset[i]['label']

                        example = image_to_tfexample(
                                image_data, b'jpg', height, width, label)

                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

# def generate_datasets(data_root):
#     train_test = np.loadtxt(os.path.join(data_root, 'train_test_split.txt'), int)
#     images_files = np.loadtxt(os.path.join(data_root, 'images.txt'), str)
#     labels = np.loadtxt(os.path.join(data_root, 'image_class_labels.txt'), int) - 1
#
#     train_dataset = []
#     test_dataset = []
#
#     for index in range(len(images_files)):
#         images_file = images_files[index, 1]
#         is_training = train_test[index, 1]
#         label = labels[index, 1]
#
#         example = {}
#         example['filename'] = os.path.join(data_root, 'images', images_file)
#         example['label'] = label
#         # example['part'] = part
#         # example['exist'] = exist
#         # example['bbox'] = bbox
#
#         if is_training:
#             train_dataset.append(example)
#         else:
#             test_dataset.append(example)
#
#     return train_dataset, test_dataset

def generate_food_datasets(data_root):
    train_dataset = []
    val_dataset = []

    train_path = os.path.join(data_root,'train')
    val_path = os.path.join(data_root,'val')

    lables = os.listdir(train_path)
    print(lables)

    for label in lables:
        image = os.listdir(train_path + label)
        for i in range(len(image)):
            example = {}
            image[i] = label + "/" + image[i]
            example['filename'] = os.path.join(train_path,image)
            example['label'] = int(label)
            train_dataset.append(example)

    for label in lables:
        image = os.listdir(val_path + label)
        for i in range(len(image)):
            example = {}
            image[i] = label + "/" + image[i]
            example['filename'] = os.path.join(val_path, image)
            example['label'] = int(label)
            val_dataset.append(example)

    return train_dataset,val_dataset

def generate_car_datasets(data_root):
    train_info = sio.loadmat(os.path.join(data_root, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
    test_info = sio.loadmat(os.path.join(data_root, 'devkit', 'cars_test_annos.mat'))['annotations'][0]

    train_dataset = []
    test_dataset = []
    label_to_class = []



    for index in range(len(train_info)):
        images_file = str(train_info['fname'][index][0])
        label = train_info['class'][index][0][0] - 1

        # labels_to_classes = {}
        # labels_to_classes[label] = get_the_class_num(train_info['class'])
        # print(images_file)

        example = {}
        example['filename'] = os.path.join(data_root, 'cars_train', images_file)
        example['label'] = int(label)
        train_dataset.append(example)
        # label_to_class.append(labels_to_classes)

    for index in range(len(test_info)):
        images_file = str(test_info['fname'][index][0])
        label = 1

        example = {}
        example['filename'] = os.path.join(data_root, 'cars_test', images_file)
        example['label'] = int(label)
        test_dataset.append(example)
    '''
    if( not has_labels(data_root)):
        for i in len(label_to_class):
            write_label_file(label_to_class[i])
    '''
    return train_dataset, test_dataset

def run(dataset_dir):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.io.gfile.exists(dataset_dir):
        tf.io.gfile.makedirs(dataset_dir)

    # Divide into train and test:
    random.seed(_RANDOM_SEED)

    # train_dataset, test_dataset = generate_car_datasets(dataset_dir)
    train_dataset, test_dataset = generate_food_datasets(dataset_dir)

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # First, convert the training and testing sets.
    convert_dataset('train', train_dataset, dataset_dir)
    convert_dataset('test', test_dataset, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the Car dataset!')

if __name__ == '__main__':
    datasets_dir = "D:/Programming/WS_DAN/datasets"
    run(datasets_dir)

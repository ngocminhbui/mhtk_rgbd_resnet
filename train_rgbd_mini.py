from resnet_train import train
from resnet_architecture import *
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('train_lst', './lists/train_mini_1.lst', 'training list')
tf.app.flags.DEFINE_string('eval_lst', './lists/eval_mini_1.lst', 'validation list')
tf.app.flags.DEFINE_string('log_dir', './log/mini_1','Directory where to write event logs and checkpoint.')

tf.app.flags.DEFINE_string('dictionary', './lists/dictionary.lst', 'dictionary')
tf.app.flags.DEFINE_string('data_dir', '/home/knmac/data/rgbd-dataset-processed-4dpng','data dir')
#tf.app.flags.DEFINE_string('data_dir', '/media/ngocminh/DATA/rgbd-dataset-processed-4dpng','data dir')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning  rate.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.app.flags.DEFINE_integer('max_steps', 500000, 'max steps')
tf.app.flags.DEFINE_boolean('resume', False, 'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,'produce fewer summaries to save HD space')
tf.app.flags.DEFINE_float('starter_learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('train_decay_rate', 0.1, 'decay rate of training phase')
tf.app.flags.DEFINE_integer('train_decay_steps', 10000, 'number of steps before decaying')


tf.app.flags.DEFINE_integer('num_classes', 10, 'number of classes')
tf.app.flags.DEFINE_integer('input_size', 224, 'width and height of image')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 16, 'number of preprocess threads')
tf.app.flags.DEFINE_integer('min_queue_examples', 20000, 'min after dequeue')

tf.app.flags.DEFINE_string('pretrained_model', './model/ResNet-L50.npy', "Path of resnet pretrained model")

''' Load list of  {filename, label_name, label_index} '''
def load_data(data_dir, data_lst):
    data = []
    train_lst = open(data_lst, 'r').read().splitlines()
    dictionary = open(FLAGS.dictionary, 'r').read().splitlines()
    for img_fn in train_lst:
        fn = os.path.join(data_dir, img_fn + '_crop.png')
        label_name = img_fn.split('/')[0]
        label_index = dictionary.index(label_name)
        data.append({
            "filename": fn,
            "label_name": label_name,
            "label_index": label_index
        })
    return data


''' Load input data using queue (feeding)'''


def read_image_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=4)

    example=tf.cast(example,tf.float32)
    ''' Image Normalization (later...) '''


    return example, label


def distorted_inputs(data_dir, data_lst):
    data = load_data(data_dir, data_lst)

    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]

    input_queue = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    # read image and label from disk
    image, label = read_image_from_disk(input_queue)

    ''' Data Augmentation '''
    image = tf.random_crop(image, [FLAGS.input_size, FLAGS.input_size, 4])
    image = tf.image.random_flip_left_right(image)

    # generate batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=FLAGS.min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=FLAGS.min_queue_examples)

    return image_batch, tf.reshape(label_batch, [FLAGS.batch_size])


def main(_):
    images, labels = distorted_inputs(FLAGS.data_dir, FLAGS.train_lst)

    is_training = tf.placeholder('bool', [], name='is_training')  # placeholder for the fusion part

    logits = inference(images,
                       num_classes=FLAGS.num_classes,
                       is_training=is_training,
                       num_blocks=[3, 4, 6, 3])
    train(is_training,logits, images, labels)


if __name__ == '__main__':
    tf.app.run(main)

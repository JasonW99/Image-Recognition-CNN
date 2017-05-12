import tensorflow as tf
import numpy as np
import os
import time

# STEPS = 100
# BATCH_SIZE = 100
# LEARNING_RATE = 0.01
# LAMBDA = 5e-4
# KEEP_RATE = 0.75

# STEPS = 150
# BATCH_SIZE = 100
# LEARNING_RATE = 0.005
# LAMBDA = 7e-4
# KEEP_RATE = 0.8

# STEPS = 300
# BATCH_SIZE = 100
# LEARNING_RATE = 0.001
# LAMBDA = 1e-3
# KEEP_RATE = 0.5

# STEPS = 300
# BATCH_SIZE = 100
# LEARNING_RATE = 0.001
# LAMBDA = 1e-2
# KEEP_RATE = 0.8

# STEPS = 100
# BATCH_SIZE = 200
# LEARNING_RATE = 0.005
# LAMBDA = 1e-1
# KEEP_RATE = 0.7

STEPS = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.01
LAMBDA = 5e-5
KEEP_RATE = 0.75

NO_CLASSES = 2
BOTTLENECK_TENSOR_SIZE = 2048
TEST_DATA_SIZE = 156


def DataProvider(tfrecods_dir):
    """Read one by one records from the given tfrecords file

    Args:
        tfrecods_dir: Full path to the tfrecords file.

    Returns:
        A set of raw image data and the corresponding label data.
    """
    filename_queue = tf.train.string_input_producer([tfrecods_dir])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_dict = {
        'class/label': tf.FixedLenFeature([], tf.int64),
        'image/encoded': tf.FixedLenFeature([], tf.string),
    }
    features = tf.parse_single_example(serialized_example, features=feature_dict)
    image = features['image/encoded']
    label = features['class/label']
    return image, label


def get_bottlenecks(sess, no_classes, batch_size, image_tensor, bottleneck_tensor, image_batch_value, label_batch_value):
    """For a batch of image data, return the corresponding bottlenecks and true labels

    Args:
        sess: The tf.Session that values will return to.
        no_classes: The number of total labels/classes. In our case, no_classes=2.
        batch_size: Size of the batch.
        image_tensor: The tensor to input the image raw data.
        bottleneck_tensor: The tensor to output the bottleneck values.
        image_batch_value: A batch of image raw data.
        label_batch_value: A batch of corresponding true labels/classes.

    Returns:
        a set of raw image data and the corresponding label data.
    """
    bottlenecks = []
    true_labels = []
    for i in range(batch_size):
        bottleneck_value = sess.run(bottleneck_tensor, feed_dict={image_tensor: image_batch_value[i]})
        bottleneck_value = np.squeeze(bottleneck_value)
        label_value = np.zeros(no_classes, dtype=np.float32)
        label_value[label_batch_value[i]] = 1.0
        bottlenecks.append(bottleneck_value)
        true_labels.append(label_value)
    return bottlenecks, true_labels


def transfer_training():
    """Implement a 'InceptionV3' transfer learning. Train a binary classifier and save the trained model to directory 'saved_model/trained_model.pb'

    Args:
        weights_init: The weights of previous trained model.
        biases_init: The biases of previous trained model.
    """
    # define the tensor of training and test data
    current_dir = os.getcwd()
    train_dir = os.path.join(current_dir, 'tfrecords/train.tfrecords')
    test_dir = os.path.join(current_dir, 'tfrecords/test.tfrecords')
    train_image, train_label = DataProvider(train_dir)
    train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=BATCH_SIZE, num_threads=4, capacity=10000, min_after_dequeue=5000)
    test_image, test_label = DataProvider(test_dir)
    test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=TEST_DATA_SIZE, num_threads=4, capacity=5000)

    # define two placeholder to hold a batch of bottlenecks and the true labels
    bottlenecks_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottlenecksInput')
    true_labels_input = tf.placeholder(tf.float32, [None, NO_CLASSES], name='TrueLabelsInput')

    # define a new fully connected layer for the transfer learning
    with tf.name_scope('transfer_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, NO_CLASSES], stddev=0.001), name='Weight')
        biases = tf.Variable(tf.zeros([NO_CLASSES]), name='Bias')
        logits = tf.add(tf.matmul(bottlenecks_input, weights), biases, name='Logits')
        logits_dropout = tf.add(tf.matmul(tf.nn.dropout(bottlenecks_input, keep_prob=KEEP_RATE), weights), biases, name='LogitsDropout')
        final_tensor = tf.nn.softmax(logits, name='ClassificationResult')

    # define the loss function
    with tf.name_scope('training_process'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits_dropout, labels=true_labels_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss_function = cross_entropy_mean + LAMBDA * tf.nn.l2_loss(weights)
        # train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss_function)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_function)

    # calculate the correct prediction rate
    with tf.name_scope('prediction_evaluation'):
        true_class = tf.argmax(true_labels_input, 1, name='TrueClass')
        pred_class = tf.argmax(final_tensor, 1, name='PredClass')
        correct_prediction = tf.equal(true_class, pred_class, name='CorrectPred')
        correct_prediction_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='CorrectPredRate')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # read the InceptionV3 model
        with tf.gfile.FastGFile(os.path.join(os.getcwd(), 'inception-v3/classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        # define two tensors as the input and output of InceptionV3 model
        image_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        # training process
        for i in range(STEPS):
            # print('step', i)
            train_image_batch_value, train_label_batch_value = sess.run([train_image_batch, train_label_batch])
            train_bottlenecks, train_true_labels = get_bottlenecks(sess, NO_CLASSES, BATCH_SIZE, image_tensor, bottleneck_tensor, train_image_batch_value, train_label_batch_value)
            # sess.run(logits, feed_dict={bottlenecks_input: train_bottlenecks, true_labels_input: train_true_labels})
            sess.run(train_step, feed_dict={bottlenecks_input: train_bottlenecks, true_labels_input: train_true_labels})

            if i % 10 == 0 or i + 1 == STEPS:
                train_accuracy = sess.run(correct_prediction_rate, feed_dict={bottlenecks_input: train_bottlenecks, true_labels_input: train_true_labels})
                print('Step %d: For %d random training samples, the training accuracy is %.2f%%' %(i, BATCH_SIZE, train_accuracy * 100))

        # calculate the prediction rate on test data
        test_image_batch_value, test_label_batch_value = sess.run([test_image_batch, test_label_batch])
        test_bottlenecks, test_true_labels = get_bottlenecks (sess, NO_CLASSES, TEST_DATA_SIZE, image_tensor, bottleneck_tensor, test_image_batch_value, test_label_batch_value)
        # print(test_true_labels)
        # print(sess.run(final_tensor, feed_dict={bottlenecks_input: test_bottlenecks}))
        test_true_class, test_pred_class,test_accuracy = sess.run([true_class, pred_class, correct_prediction_rate], feed_dict={bottlenecks_input: test_bottlenecks, true_labels_input: test_true_labels})
        print('\nFor %d test samples, the overall test accuracy is %.2f%%.\n'
              'Among %d POSITIVE samples, the Sensitivity (true positive rate) is %.2f%%.\n'
              'Among %d NEGATIVE samples, the Specificity (true negative rate) is %.2f%%.\n'
              'Among %d POSITIVE predictions, the Positive Predictive Value is %.2f%%.\n'
              'Among %d NEGATIVE predictions, the Negative Predictive Value is %.2f%%.' %
              (TEST_DATA_SIZE, test_accuracy * 100,
               sum(test_true_class), sum((test_true_class == 1) & (test_pred_class == 1)) / sum(test_true_class) * 100,
               sum(test_true_class == 0), sum((test_true_class == 0) & (test_pred_class == 0)) / sum(test_true_class == 0) * 100,
               sum(test_pred_class), sum((test_true_class == 1) & (test_pred_class == 1)) / sum(test_pred_class) * 100,
               sum(test_pred_class == 0), sum((test_true_class == 0) & (test_pred_class == 0)) / sum(test_pred_class == 0) * 100,))

        coord.request_stop()
        coord.join(threads)

        # save the trained model
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['transfer_training_ops/ClassificationResult','prediction_evaluation/CorrectPredRate', 'pool_3/_reshape'])
        # saved_model_dir = os.path.join(current_dir, 'saved_model')
        # os.makedirs(saved_model_dir)
        with tf.gfile.GFile("saved_model/trained_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    t1 = time.time()
    transfer_training()
    print((time.time() - t1)/60)


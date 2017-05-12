import tensorflow as tf
import numpy as np
import os
import time

NO_CLASSES = 2
TEST_DATA_SIZE = 200


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


def dataset_test(model_dir, test_dataset_dir):
    """Use the trained model to classify the given dataset and compare the results with the true labels ('Negative' or 'Positive').

    Args:
        model_dir: Full path to the saved model.
        test_dataset_dir: Full path to the target dataset.
    """
    start = time.time()
    test_image, test_label = DataProvider(dataset_dir)
    test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=TEST_DATA_SIZE, num_threads=4, capacity=5000)

    print(time.time()-start)
    with tf.Session() as sess:
        with tf.gfile.FastGFile(model_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        image_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')  # define the tensor for the raw graph
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')  # define the tensor holding 2048 features as the bottleneck
        bottlenecks_input = sess.graph.get_tensor_by_name('BottlenecksInput:0')  # define the tensor for bottleneck input
        true_labels_input = sess.graph.get_tensor_by_name('TrueLabelsInput:0')  # define the tensor for true class/label input
        true_class = sess.graph.get_tensor_by_name('prediction_evaluation/TrueClass:0')  # define the tensor for true class/label input
        pred_class = sess.graph.get_tensor_by_name('prediction_evaluation/PredClass:0') # define the tensor for predicted class/label input
        correct_prediction_rate = sess.graph.get_tensor_by_name('prediction_evaluation/CorrectPredRate:0') # define the tensor for correct prediction rate

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        test_image_batch_value, test_label_batch_value = sess.run([test_image_batch, test_label_batch])
        test_bottlenecks, test_true_labels = get_bottlenecks(sess, NO_CLASSES, TEST_DATA_SIZE, image_tensor, bottleneck_tensor, test_image_batch_value, test_label_batch_value)
        test_true_class, test_pred_class, test_accuracy = sess.run([true_class, pred_class, correct_prediction_rate],
                                                                   feed_dict={bottlenecks_input: test_bottlenecks, true_labels_input: test_true_labels})

        print(time.time() - start)
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
        print(time.time() - start)









current_dir = os.getcwd()
# model_dir = os.path.join(current_dir, 'saved_model/trained_model.pb')
model_dir = os.path.join(current_dir, 'saved_model/trained_model_V1_2586p-2371n.pb')
# model_dir = os.path.join(current_dir, 'saved_model/trained_model_V2.pb')
# model_dir = os.path.join(current_dir, 'saved_model/trained_model(BS-100-Step-700-LR-0.01-Lamba-5e-4-KR-0.75).pb')
dataset_dir = os.path.join(current_dir, 'tfrecords/test.tfrecords')

dataset_test(model_dir, dataset_dir)


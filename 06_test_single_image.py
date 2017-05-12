import tensorflow as tf
import os


def single_image_test(model_dir, image_dir):
    """Use the trained model to classify one given image into 'Negative' or 'Positive' group.

    Args:
        model_dir: Full path to the saved model.
        image_dir: Full path to the target image.
    """
    with tf.Session() as sess:
        # read the trained model in to Session
        with tf.gfile.FastGFile(model_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        # print([m.values() for m in sess.graph.get_operations()]) # may use this line to check all the tensor names in the loaded model
        image_tensor = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')  # define the tensor for the raw graph
        bottleneck_tensor = sess.graph.get_tensor_by_name('pool_3/_reshape:0')  # define the tensor holding 2048 features as the bottleneck
        bottlenecks_input = sess.graph.get_tensor_by_name('BottlenecksInput:0')  # define the tensor for bottleneck input
        classification_result = sess.graph.get_tensor_by_name('transfer_training_ops/ClassificationResult:0')  # define the tensor for the GLM result

        image = tf.gfile.FastGFile(image_dir, 'rb').read()  # read the image into raw data
        bottlenecks_input_value = sess.run(bottleneck_tensor, {image_tensor: image})  # compute bottleneck value
        classification_result_value = sess.run(classification_result, {
            bottlenecks_input: bottlenecks_input_value})  # compute classification result
        print(
            'With probability %.2f%% the test image belongs to group NEGATIVE\nWith probability %.2f%% the test image belongs to group POSITIVE' % (
            classification_result_value[0, 0] * 100, classification_result_value[0, 1] * 100))







current_dir = os.getcwd()

# model_dir = os.path.join(current_dir, 'saved_model/trained_model.pb')
# model_dir = os.path.join(current_dir, 'saved_model/trained_model(BS-100-Step-2000-LR-0.01-Lamba-5e-4-KR-0.75).pb')
model_dir = os.path.join(current_dir, 'saved_model/trained_model_V1_2586p-2371n.pb')
# image_dir = os.path.join(current_dir, 'dataset/Train/Negative/14240590_167886546951764_854605361_n.jpg')
image_dir = os.path.join(current_dir, 'dataset/Test/Positive/na1386111633824.jpg')
# image_dir = os.path.join(current_dir, 'test_eval/3.jpg')
single_image_test(model_dir, image_dir)





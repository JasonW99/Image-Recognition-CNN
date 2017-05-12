import glob
import os
import tensorflow as tf
import numpy as np


IMAGE_DEPTH = 3

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(dataset_dir, output_file_dir):
    """Generates TFRecord for images.

    Args:
        dataset_dir: Path to dataset directory. In pur case, this directory will hold two subdirectories with name "Negative" and "Positive".
        output_file_path: Full path to output TFRecord file.
    """
    # create a 'key.txt' file to record 'class_label' and 'class_name' for future reference
    class_names = os.listdir(dataset_dir)
    np.savetxt('key.txt', np.append([np.arange(len(class_names))], [class_names], axis=0).T, fmt='%s', newline='\r\n')

    # define the writer and the writing directory
    writer = tf.python_io.TFRecordWriter(output_file_dir)

    # initialize the 'class_label' (in our case, class_label for "Negative" will be 0, class_label for "Positive" will be 1)
    class_label = 0

    for class_name in class_names:
        image_files = glob.glob(os.path.join(dataset_dir, class_name, '*.jpg'))
        print(image_files)
        for image_dir in image_files:
            raw_image = tf.gfile.FastGFile(image_dir, "r").read()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(raw_image), # string, JPEG encoding of RGB image
                'class/text': _bytes_feature(tf.compat.as_bytes(class_name)), # "Negative" or "Positive"
                'class/label': _int64_feature(class_label)  # "0" or "1"
            }))
            writer.write(example.SerializeToString())
        class_label = class_label + 1
    writer.close()






current_dir = os.getcwd()
tfrecords_dir = os.path.join(current_dir, 'tfrecords')
# os.makedirs(tfrecords_dir)

convert_to_tfrecord(os.path.join(current_dir, 'dataset/Train'), os.path.join(tfrecords_dir,'train.tfrecords'))
convert_to_tfrecord(os.path.join(current_dir, 'dataset/Test'), os.path.join(tfrecords_dir,'test.tfrecords'))







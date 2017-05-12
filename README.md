# Image Recognition Project (CNN)
assignment: train a binary classifier using CNN approach (transfer learning based on 'InceptionV3' model)

## Directory Structure
for the complete project, there should be 6 python files and 4 sub-folders in the directory.
* '01_generate_tfrecords.py': this file generate '.tfrecords' file from given image data.
* '02_transferlearning.py': this file train the model.
* '03_transferlearning(continous training).py': this file train the model continuously from previous trained results.
* '04_test_given_dataset.py': this file classify images from .tfrecords using the trained model.
* '05_test_folder_image.py':  this file classify images within given folder using the trained model.
* '06_test_single_image.py':  this file classify a single image using the trained model.
* 'dataset': this folder contains image data.
* 'tfrecords': this folder contains '.tfrecords' files, which are generated from image data.
* 'inception-v3': this folder contains 'InceptionV3' model.
* 'saved_model': this folder contains trained model.

## Generate tfrecords
use '01_generate_tfrecords.py' to generate the training and test dataset.
'.tfrecords' is the standard tensorflow data format. to generate a desired dataset from pictures, we need to allocate '.jpg' files in the 'dataset' sub-folder as following:
```bash
#
# dataset/Train/Negative/hksafjkdk.jpg
# dataset/Train/Negative/hksdsfsdfsffff.jpg
# dataset/Train/Negative/hkdd.jpg
# ......
# ......
# dataset/Train/Positive/hkrrdk.jpg
# dataset/Train/Positive/hkvb.jpg
# dataset/Train/Positive/hkr456452656dk.jpg
# ......
# ......
# dataset/Test/Negative/hksafjkdk.jpg
# dataset/Test/Negative/hksdsfsdfsffff.jpg
# dataset/Test/Negative/hkdd.jpg
# ......
# ......
# dataset/Test/Positive/hkrrdk.jpg
# dataset/Test/Positive/hkvb.jpg
# dataset/Test/Positive/hkr456452656dk.jpg
# ......
# ......

```

## Train The Model
using the 'InceptionV3' model we can generate a 2048-dimension bottleneck for the given image. then we construct a new fully connected layer(FC layer) leading the bottleneck to the binary classification result.
the training data may help us to optimize the coefficients on the FC layer, and the optimized results establish the model we need. during the training process, we implement dropout and L2-regularization(ridge regression) to prevent overfitting.
a newly trained model will be save to directory 'saved_model/trained_model.pb'. currently, we have already trained a model using 395 NEGATIVE data and 396 POSITIVE data. the model is saved as 'saved_model/trained_model(BS-100-Step-700-LR-0.01-Lamba-5e-4-KR-0.75).pb'.
the training and test result is like following:
```bash
#
# Step 0: For 100 random training samples, the training accuracy is 55.00%
# Step 50: For 100 random training samples, the training accuracy is 72.00%
# Step 100: For 100 random training samples, the training accuracy is 96.00%
# ...
# ...
# Step 650: For 100 random training samples, the training accuracy is 99.00%
# Step 699: For 100 random training samples, the training accuracy is 98.00%
#
# For 100 test samples, the overall test accuracy is 80.00%.
# Among 50 POSITIVE samples, the Sensitivity (true positive rate) is 84.00%.
# Among 50 NEGATIVE samples, the Specificity (true negative rate) is 76.00%.
# Among 54 POSITIVE predictions, the Positive Predictive Value is 77.78%.
# Among 46 NEGATIVE predictions, the Negative Predictive Value is 82.61%.
#
```

## Train Continuously
use '03_transferlearning(continous training).py' to train the model continuously.
to prevent repeatedly training a model from very beginning, we may use the file to get the important coefficients of a previous trained model, and use those obtained values to initialize the coefficients in a new training process.
again, a newly trained model will be save to directory 'saved_model/trained_model.pb'.


## Test tfrecords
use '04_test_given_dataset.py' to classify a batch of images recorded by tfreords.
the dataset should be in '.tfrecords' format, and the output is like following:
```bash
#
# For 100 test samples, the overall test accuracy is 80.00%.
# Among 50 POSITIVE samples, the Sensitivity (true positive rate) is 84.00%.
# Among 50 NEGATIVE samples, the Specificity (true negative rate) is 76.00%.
# Among 54 POSITIVE predictions, the Positive Predictive Value is 77.78%.
# Among 46 NEGATIVE predictions, the Negative Predictive Value is 82.61%.
#
```

## Test Image Folder
use '05_test_folder_image.py' to classify images within the given folder.
the output is like following:
```
# C:\***\dataset\Test\Negative\10.jpg
# With probability 94.62% the test image belongs to group NEGATIVE
# With probability 5.38% the test image belongs to group POSITIVE
# C:\***\dataset\Test\Negative\11.jpg
# With probability 95.90% the test image belongs to group NEGATIVE
# With probability 4.10% the test image belongs to group POSITIVE
# ...
#
```

## Test Single Image
use '06_test_single_image.py' to classify a single image.
the output is like following:
```bash
#
# With probability 1.42% the test image belongs to group NEGETIVE
# With probability 98.58% the test image belongs to group POSITIVE
#
```

## Authors

* **Jason W**




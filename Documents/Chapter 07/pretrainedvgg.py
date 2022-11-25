import sys
sys.path.append("/tensorflow/models/slim")

from datasets imoprt datasets_utils
import tensorflow as tensorflow
target_dir = '<some_path> + vgg/vgg_checkpoints'
# download checkpoints

import urllib2

# image from my old website
url = ("http://xxx.xxx.xxx.xxx/wp-content/uploads/2017/07/Screen-Shot-2017-06-19-at-18.08.23-1-e1499426621151.png")
im_as_string = urllib2.urlopen(url).read()
im = tf.image.decode_png(im_as_string, channels=3)

# We would use this if we wanted to load a file from the computer
# filename_queue = tf.train.string_input_producter(tf.train.match_filenames_once("./images/*.jpg"))
# image_reader = tf.WholeFileReader()
# _, image_file = image_reader.read(filename_queue)
# image = tf.image.decode)jpeg(image_file)

from nets import vgg
image_size = vgg.vgg_16.default_image_size

from preprocessing import vgg_preprocessing
processed_im = vgg_preprocessing.preprocess_image(image, image_size, image_)

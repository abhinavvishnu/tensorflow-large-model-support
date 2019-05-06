# Copyright 2018 IBM All Rights Reserved.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This model uses the ResNet50 from keras_applications to demonstrate
# how to enable TensorFlow Large Model Support (TFLMS) in a Keras model that
# cannot fit in GPU memory when using larger resolution data. To simplify
# the running of the model with different higher resolution images,
# the code uses a random image generator to generate synthetic image data.
#
# This model provides a convenient way to test out the capabilities
# of TFLMS. Command line parameters allow the user to change the size of
# the input image data, enable or disable TFLMS, set TFLMS tunables,
# and enable or disable CUDA profiling to ease collection of profile data.
#
# This model uses randomly generated synthetic image data. The random
# generation is intentionally not using GPU operations to avoid impacting
# the memory usage profile of the model. Since random generated images
# are being used this code should not be used as an example on how to
# correctly and efficiently pipeline data using datasets, etc.
#
# Point in time test results:
# Environment:
#   IBM AC922 with NVIDIA Volta V100 GPUs containing 16GB of RAM
#   TensorFlow 1.12
#   NVIDIA CUDA 10.0.130
#   NVIDIA cuDNN 7.3.1
#
# Max resolution without TFLMS: 2300x2300
# Max resolution with TFLMS: 3900x3900
# This is a 2.88x resolution increase with TFLMS.
#
# Invocation examples:
# Run without LMS:
#   python Keras_ResNet50.py --image_size 2300
# Run with LMS:
#   python Keras_ResNet50.py --image_size 3900 --lms
# Swap some, but not all tensors:
#  python Keras_ResNet50.py --image_size 2400 --lms --n_tensors 20 --lb 30


import argparse
import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow_large_model_support import LMSKerasCallback

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2

import time
import ctypes
#_cudart = ctypes.CDLL('libcudart.so')

tf.logging.set_verbosity(tf.logging.INFO)


class CudaProfileCallback(Callback):
    def __init__(self, profile_epoch, profile_batch_start, profile_batch_end):
        self._epoch = profile_epoch - 1
        self._start = profile_batch_start
        self._end = profile_batch_end
        self.epoch_keeper = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_keeper = epoch
    def on_batch_begin(self, batch, logs=None):
        if batch == self._start and self.epoch_keeper == self._epoch:
            print('Starting cuda profiler')
#            _cudart.cudaProfilerStart()
        if batch == self._end and self.epoch_keeper == self._epoch:
            print('Stopping cuda profiler')
#            _cudart.cudaProfilerStop()

def random_image_generator(batch_size, num_classes, input_shape):
    # This generator yields batches of randomly generated images and categories.
    # The random generation parts came from
    # https://github.com/tensorflow/tensorflow/blob/v1.12.0/tensorflow/python/keras/testing_utils.py#L29
    # These two random generations take a long time for large dimenstions and should
    # really be in the while loop below to have
    # better random images being generated for every yield. They were moved out of the while loop
    # to speed up the generator since the intent of this example is to show resolutions with
    # and without TFLMS versus a good data spread and loss / accuracy numbers.
    templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
    random_data = np.random.normal(loc=0, scale=1., size=input_shape)
    while True:
        y = np.random.randint(0, num_classes, size=(batch_size,))
        x = np.zeros((batch_size,) + input_shape, dtype=np.float32)
        for i in range(batch_size):
            x[i] = templates[y[i]] + random_data
        x_array = np.array(x)
        y_array = tf.keras.utils.to_categorical(y, num_classes)
        yield(x_array, y_array)

def get_callbacks(args):
    callbacks = []

    if args.nvprof:
        callbacks.append(CudaProfileCallback(args.nvprof_epoch,
                                             args.nvprof_start,
                                             args.nvprof_stop))

    # Enable TFLMS
    if args.lms:
        # Specifying this starting name, from previous runs of LMS,
        # speeds up graph analysis time.
        starting_names = ['bn_conv1/cond/pred_id']
        lms = LMSKerasCallback(n_tensors=args.n_tensors, lb=args.lb,
                               starting_op_names=starting_names)
        callbacks.append(lms)

    return callbacks
"""
def run_model(args):
    # Configure the memory optimizer
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
    K.set_session(tf.Session(config=config))

    image_dim = args.image_size
    input_shape = (image_dim, image_dim, 3)

    num_classes = args.num_classes
    batch_size = args.batch_size

    resnet50 = tf.keras.applications.ResNet50(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    resnet50.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    random_generator = random_image_generator(batch_size, num_classes,
                                              input_shape)
    resnet50.fit_generator(random_generator, steps_per_epoch=args.steps,
                           epochs=args.epochs, callbacks=get_callbacks(args))

"""
def execute_model(model, input_shape):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    random_generator = random_image_generator(args.batch_size, args.num_classes,
                                              input_shape)
    # This is the warmup phase
    model.fit_generator(random_generator, verbose=1, steps_per_epoch=args.steps,
                           epochs=1, callbacks=get_callbacks(args))

    # this is the actual timing calculation


    start = time.time()
    model.fit_generator(random_generator, verbose=1, steps_per_epoch=args.steps,
                           epochs=args.epochs, callbacks=get_callbacks(args))
    end = time.time()
    print('images-per-second:') 
    print((args.batch_size * args.epochs * args.steps)/(end - start))

def run_model(args):
    # Configure the memory optimizer
    config = tf.ConfigProto()
    config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
    K.set_session(tf.Session(config=config))

    num_classes = args.num_classes
    batch_size = args.batch_size

    model_name = args.model
    if model_name == 'ResNet50':
        model = ResNet50(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'ResNet101':
        model = tensorflow.python.keras.applications.resnet.ResNet101(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'VGG16':
        model = VGG16(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'VGG19':
        model = VGG19(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'Xception':
        model = Xception(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'MobileNet':
        model = MobileNet(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'MobileNetV2':
        model = MobilenetV2(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    elif model_name == 'InceptionV3':
        model = InceptionV3(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    else:
        print('Running with ResNet50 -- the default model')
        model = ResNet50(weights=None, include_top=True,
                                              input_shape=input_shape,
                                              classes=num_classes)
    execute_model(model, input_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        default=1,
                        help='Number of epochs to run. (Default 1)')
    parser.add_argument("--steps", type=int,
                        default=4,
                        help='Number of steps per epoch. (Default 10)')
    parser.add_argument("--image_size", type=int,
                        default=256,
                        help='Dimension of one side of the square image '
                             'to be generated. (Default 500)')
    # LMS parameters
    lms_group = parser.add_mutually_exclusive_group(required=False)
    lms_group.add_argument('--lms', dest='lms', action='store_true',
                           help='Enable TFLMS')
    lms_group.add_argument('--no-lms', dest='lms', action='store_false',
                           help='Disable TFLMS (Default)')
    parser.set_defaults(lms=False)
    parser.add_argument("--n_tensors", type=int,
                        default=-1,
                        help='The number of tensors to swap. Default -1 (all)')
    #AMD Change
    parser.add_argument("--batch_size", type=int,
                        default=32,
                        help='batch size. Default 1 (all)')
    parser.add_argument("--num_classes", type=int,
                        default=10,
                        help='Number of classes in the model. Default 10 (all)')
    parser.add_argument("--autotune_image_size", type=int,
                        default=0,
                        help='Auto-tune the Image size')
    parser.add_argument("--model", type=str,
                        default="ResNet50",
                        help='Deep neural network model. Default ResNet50. Please check'
                        ' keras.applications for supported models')

    parser.add_argument("--lb", type=int,
                        default=1,
                        help='Lowerbound value for LMS. A tensor will be '
                             'swapped in during the backward phase at least lb '
                             'nodes before it in the graph. Default 1.')

    parser.set_defaults(nvprof=False)

    args = parser.parse_args()

    image_dim = args.image_size
    input_shape = (image_dim, image_dim, 3)

    run_model(args)

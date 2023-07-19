import logging
import os
import string
from glob import glob
import cv2
import numpy as np
import tensorrt as trt
from numba import cuda

import constants as constants

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


import os
import string
import numpy as np
import tensorrt as trt
from numba import cuda
import constants
class RecognitionEngine(object):
    __ENGINE_NAME = 'recognizer.engine'
    __MODELS_PATH = "./models/recognition_weights.np"
    def __init__(self, max_batch_size=2):
        self.ALPHABET = '-' + string.digits + string.ascii_lowercase
        self.BLANK_INDEX = 0
        self.SEQUENCE_SIZE = 30
        self.ALPHABET_SIZE = len(self.ALPHABET)
        self.HIDDEN_SIZE = 256
        self.OUTPUT_SHAPE = [self.SEQUENCE_SIZE, self.ALPHABET_SIZE]
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.max_batch_size = max_batch_size
        if os.path.exists(RecognitionEngine.__ENGINE_NAME):
            with open(RecognitionEngine.__ENGINE_NAME, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            print("Model loaded")
        else:
            self.engine = self.__create_inference_engine(np.fromfile(RecognitionEngine.__MODELS_PATH, dtype=np.float32))
            with open(RecognitionEngine.__ENGINE_NAME, 'wb') as f:
                f.write(self.engine.serialize())
            print("Model built")
    def predict(self, imgs):
        cuda_stream = cuda.stream()
        batch_size = imgs.shape[0]
        softmax_dims = np.ones((batch_size, self.SEQUENCE_SIZE, 1), dtype=np.int32) * self.ALPHABET_SIZE
        output = np.empty([batch_size] + self.OUTPUT_SHAPE, dtype=np.float32)
        cuda_imgs = cuda.to_device(imgs, cuda_stream)
        cuda_dims = cuda.to_device(softmax_dims, cuda_stream)
        cuda_output = cuda.to_device(output, cuda_stream)
        bindings = [cuda_imgs.device_ctypes_pointer.value, cuda_dims.device_ctypes_pointer.value,
                    cuda_output.device_ctypes_pointer.value]
        with self.engine.create_execution_context() as execution_context:
            execution_context.execute_async(batch_size, bindings, cuda_stream.handle.value, None)
        cuda_stream.synchronize()
        cuda_output.copy_to_host(output, stream=cuda_stream)
        raw_labels = output.argmax(2)
        raw_probs = output.max(2)
        labels = []
        probs = []
        for i in range(batch_size):
            current_prob = 1.0
            current_label = []
            for j in range(self.SEQUENCE_SIZE):
                if raw_labels[i][j] != self.BLANK_INDEX and not (j > 0 and raw_labels[i][j] == raw_labels[i][j - 1]):
                    current_label.append(self.ALPHABET[raw_labels[i][j]])
                    current_prob *= raw_probs[i][j]
            if not current_label:
                current_label.append(self.ALPHABET[self.BLANK_INDEX])
                current_prob = 0.0
            labels.append(''.join(current_label))
            probs.append(current_prob)
        return labels, probs
    def __create_inference_engine(self, weights):
        with trt.Builder(
                self.trt_logger) as builder, builder.create_network() as network, builder.create_builder_config() as builder_config:
            if builder.platform_has_fast_fp16:
                print("fp 16 are using")
                builder_config.set_flag(trt.BuilderFlag.FP16)
            input_layer = network.add_input('data', trt.DataType.FLOAT, (constants.IMG_C,
                                                                         constants.RECOGNIZER_IMAGE_H,
                                                                         constants.RECOGNIZER_IMAGE_W))
            kernel_size = (3, 3)
            stride = (1, 1)
            channels = [128, 256, 512]
            index = 0
            prev_layer = input_layer
            for i in range(3):
                for j in range(3):
                    conv_weights_count = prev_layer.shape[0] * channels[i] * kernel_size[0] * kernel_size[1]
                    conv_weights = weights[index:index + conv_weights_count]
                    index += conv_weights_count
                    conv_biases_count = channels[i]
                    conv_biases = weights[index:index + conv_biases_count]
                    index += conv_biases_count
                    conv_layer = network.add_convolution(prev_layer, channels[i], kernel_size, conv_weights,
                                                         conv_biases)
                    conv_layer.stride = stride
                    if i == 2 and j == 2:
                        conv_layer.padding = (0, 0)
                    else:
                        conv_layer.padding = (1, 1)
                    scale = weights[index:index + channels[i]]
                    index += channels[i]
                    bias = weights[index:index + channels[i]]
                    index += channels[i]
                    mean = weights[index:index + channels[i]]
                    index += channels[i]
                    var = weights[index:index + channels[i]]
                    index += channels[i]
                    combined_scale = scale / np.sqrt(var + 1e-5)
                    combined_bias = bias - mean * combined_scale
                    bn = network.add_scale(conv_layer.get_output(0), trt.ScaleMode.CHANNEL,
                                           combined_bias,
                                           combined_scale,
                                           np.ones_like(combined_bias))
                    activation = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)
                    prev_layer = activation.get_output(0)
                if i < 2:
                    pooling = network.add_pooling(prev_layer, trt.PoolingType.MAX, (2, 2))
                    pooling.stride = (2, 2)
                    prev_layer = pooling.get_output(0)
            shuffle = network.add_shuffle(prev_layer)
            shuffle.reshape_dims = (prev_layer.shape[0] * prev_layer.shape[1], prev_layer.shape[2])
            shuffle.second_transpose = (1, 0)
            embedding_shape = (self.HIDDEN_SIZE, shuffle.get_output(0).shape[1])
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]
            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(
                shuffle.get_output(0),
                trt.MatrixOperation.NONE,
                embedding.get_output(0),
                trt.MatrixOperation.TRANSPOSE
            )
            embedding_bias_shape = (1, self.HIDDEN_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]
            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)
            activation = network.add_activation(add_bias.get_output(0), trt.ActivationType.RELU)
            lstm_input_size = self.HIDDEN_SIZE
            lstm = network.add_rnn_v2(activation.get_output(0),
                                      1, self.HIDDEN_SIZE, self.SEQUENCE_SIZE, trt.RNNOperation.LSTM)
            gates = [trt.RNNGateType.INPUT, trt.RNNGateType.FORGET, trt.RNNGateType.CELL, trt.RNNGateType.OUTPUT]
            input_weights = weights[index:index + lstm_input_size * len(gates) * self.HIDDEN_SIZE]
            index += lstm_input_size * len(gates) * self.HIDDEN_SIZE
            rec_weights = weights[index:index + len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE]
            index += len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE
            input_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
            index += self.HIDDEN_SIZE * len(gates)
            rec_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
            index += self.HIDDEN_SIZE * len(gates)
            hidden_2 = self.HIDDEN_SIZE ** 2
            hidden_2_2 = lstm_input_size * self.HIDDEN_SIZE
            for i in range(len(gates)):
                lstm.set_weights_for_gate(0, gates[i], True,
                                          input_weights[i * hidden_2_2:i * hidden_2_2 + hidden_2_2])
                lstm.set_weights_for_gate(0, gates[i], False,
                                          rec_weights[i * hidden_2:i * hidden_2 + hidden_2])
                lstm.set_bias_for_gate(0, gates[i], True,
                                       input_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
                lstm.set_bias_for_gate(0, gates[i], False,
                                       rec_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
            ### LAST LINEAR LAYER
            embedding_shape = (self.ALPHABET_SIZE, self.HIDDEN_SIZE)
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]
            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(
                lstm.get_output(0),
                trt.MatrixOperation.NONE,
                embedding.get_output(0),
                trt.MatrixOperation.TRANSPOSE
            )
            embedding_bias_shape = (1, self.ALPHABET_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]
            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)
            dimensions = network.add_input('dimensions', trt.DataType.INT32, (self.SEQUENCE_SIZE, 1))
            softmax = network.add_ragged_softmax(add_bias.get_output(0), dimensions)
            builder.max_batch_size = self.max_batch_size
            builder_config.max_workspace_size = 1 << 30
            network.mark_output(softmax.get_output(0))
            engine = builder.build_engine(network, builder_config)
            return engine



class RecognitionEngine_UAE(object):
    __ENGINE_NAME__ = 'extended_recognizer.engine'

    def __init__(self, max_batch_size=1, create_engine=False):
        self.ALPHABET = '-' + string.digits + string.ascii_lowercase
        self.BLANK_INDEX = 0

        self.SEQUENCE_SIZE = 38
        self.ALPHABET_SIZE = len(self.ALPHABET)
        self.HIDDEN_SIZE = 256

        self.OUTPUT_SHAPE = [self.SEQUENCE_SIZE, self.ALPHABET_SIZE]
        self.OUTPUT_2_SHAPE = [constants.EXTENDED_RECOGNIZER_OUTPUT_2_SIZE]
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.max_batch_size = max_batch_size
        self.weights_path = "/opt/api/models/recognizer_weights_extended_uae.np"
        self.engine_path = "/opt/api/models/engine/extended_recognizer.engine"
        if create_engine:
            if os.path.exists(self.engine_path):
                os.remove(self.engine_path)
                logging.debug("LP Detection old engine removed")
            self.engine = self.__create_inference_engine(np.fromfile(self.weights_path, dtype=np.float32))
            with open(self.engine_path, 'wb') as f:
                f.write(self.engine.serialize())
        else:
            with open(self.engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        self.execution_context = self.engine.create_execution_context()
        logging.info("LP Base Recognizer engine initialized")

    def predict(self, imgs):
        cuda_stream = cuda.stream()

        batch_size = imgs.shape[0]

        softmax_dims = np.ones((batch_size, self.SEQUENCE_SIZE, 1), dtype=np.int32) * self.ALPHABET_SIZE

        output_2 = np.empty([batch_size] + self.OUTPUT_2_SHAPE, dtype=np.float32)
        output = np.empty([batch_size] + self.OUTPUT_SHAPE, dtype=np.float32)

        cuda_imgs = cuda.to_device(imgs, cuda_stream)
        cuda_dims = cuda.to_device(softmax_dims, cuda_stream)
        cuda_output_2 = cuda.to_device(output_2, cuda_stream)
        cuda_output = cuda.to_device(output, cuda_stream)

        bindings = [cuda_imgs.device_ctypes_pointer.value, cuda_dims.device_ctypes_pointer.value,
                    cuda_output_2.device_ctypes_pointer.value,cuda_output.device_ctypes_pointer.value]
        self.execution_context.execute_async(batch_size, bindings, cuda_stream.handle.value, None)
        cuda_stream.synchronize()

        cuda_output_2.copy_to_host(output_2, cuda_stream)
        cuda_output.copy_to_host(output, cuda_stream)
        raw_labels = output.argmax(2)
        raw_probs = output.max(2)
        labels = []
        probs = []
        for i in range(batch_size):
            current_prob = 1.0
            current_label = []
            for j in range(self.SEQUENCE_SIZE):

                if raw_labels[i][j] != self.BLANK_INDEX and not (j > 0 and raw_labels[i][j] == raw_labels[i][j - 1]):
                    current_label.append(self.ALPHABET[raw_labels[i][j]])
                    current_prob *= raw_probs[i][j]

            if not current_label:
                current_label.append(self.ALPHABET[self.BLANK_INDEX])
                current_prob = 0.0
            labels.append(''.join(current_label))
            probs.append(current_prob)

        return labels, probs

    def __create_inference_engine(self, weights):

        with trt.Builder(
                self.trt_logger) as builder, builder.create_network() as network, builder.create_builder_constants() as builder_constants:
            input_layer = network.add_input('data', trt.DataType.FLOAT, (
                constants.IMG_C, constants.EXTENDED_RECOGNIZER_IMAGE_HEIGHT, constants.EXTENDED_RECOGNIZER_IMAGE_WIDTH))

            kernel_size = (3, 3)
            stride = (1, 1)
            channels = [128, 256, 512]
            index = 0

            prev_layer = input_layer

            for i in range(3):
                for j in range(3):
                    conv_weights_count = prev_layer.shape[0] * channels[i] * kernel_size[0] * kernel_size[1]
                    conv_weights = weights[index:index + conv_weights_count]
                    index += conv_weights_count

                    conv_biases_count = channels[i]
                    conv_biases = weights[index:index + conv_biases_count]
                    index += conv_biases_count

                    conv_layer = network.add_convolution(prev_layer, channels[i], kernel_size, conv_weights,
                                                         conv_biases)
                    conv_layer.stride = stride

                    if i == 2 and j == 2:
                        conv_layer.padding = (0, 0)
                    else:
                        conv_layer.padding = (1, 1)

                    scale = weights[index:index + channels[i]]
                    index += channels[i]

                    bias = weights[index:index + channels[i]]
                    index += channels[i]

                    mean = weights[index:index + channels[i]]
                    index += channels[i]

                    var = weights[index:index + channels[i]]
                    index += channels[i]

                    combined_scale = scale / np.sqrt(var + 1e-5)
                    combined_bias = bias - mean * combined_scale

                    bn = network.add_scale(conv_layer.get_output(0), trt.ScaleMode.CHANNEL, combined_bias,
                                           combined_scale, np.ones_like(combined_bias))

                    activation = network.add_activation(bn.get_output(0), trt.ActivationType.RELU)
                    prev_layer = activation.get_output(0)

                if i < 2:
                    pooling = network.add_pooling(prev_layer, trt.PoolingType.MAX, (2, 2))
                    pooling.stride = (2, 2)
                    prev_layer = pooling.get_output(0)
            shuffle = network.add_shuffle(prev_layer)
            shuffle.reshape_dims = (prev_layer.shape[0] * prev_layer.shape[1], prev_layer.shape[2])
            shuffle.second_transpose = (1, 0)
            embedding_shape = (self.HIDDEN_SIZE, shuffle.get_output(0).shape[1])
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]
            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(shuffle.get_output(0), trt.MatrixOperation.NONE,
                                                                embedding.get_output(0), trt.MatrixOperation.TRANSPOSE)

            embedding_bias_shape = (1, self.HIDDEN_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]
            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)
            activation = network.add_activation(add_bias.get_output(0), trt.ActivationType.RELU)

            newShuffle = network.add_shuffle(activation.get_output(0))
            newShuffle.reshape_dims = (activation.get_output(0).shape[0] * activation.get_output(0).shape[1], 1, 1)

            countryClsWeights = weights[index:index + 9728 * 7]
            index += 9728 * 7

            countryClsBias = weights[index:index + 7]
            index += 7

            countryLayer = network.add_fully_connected(newShuffle.get_output(0), 7, countryClsWeights, countryClsBias)

            lstmLayerCount = 2
            lstm_input_size = self.HIDDEN_SIZE

            lstm = network.add_rnn_v2(activation.get_output(0), lstmLayerCount, self.HIDDEN_SIZE, self.SEQUENCE_SIZE,
                                      trt.RNNOperation.LSTM)
            lstm.direction = trt.RNNDirection.BIDIRECTION
            gates = [trt.RNNGateType.INPUT, trt.RNNGateType.FORGET, trt.RNNGateType.CELL, trt.RNNGateType.OUTPUT]

            for layerIndex in range(lstmLayerCount):

                rela_index = 2 * layerIndex

                if layerIndex == 1:
                    iw_count = lstm_input_size * lstm_input_size * 2
                else:
                    iw_count = lstm_input_size * lstm_input_size

                input_weights = weights[index:index + len(gates) * iw_count]
                index += iw_count * len(gates)
                rec_weights = weights[index:index + len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE]
                index += len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE
                input_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
                index += self.HIDDEN_SIZE * len(gates)
                rec_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
                index += self.HIDDEN_SIZE * len(gates)
                hidden_2 = self.HIDDEN_SIZE ** 2

                for i in range(len(gates)):
                    lstm.set_weights_for_gate(rela_index, gates[i], True,
                                              input_weights[i * iw_count:i * iw_count + iw_count])
                    lstm.set_weights_for_gate(rela_index, gates[i], False,
                                              rec_weights[i * hidden_2:i * hidden_2 + hidden_2])
                    lstm.set_bias_for_gate(rela_index, gates[i], True,
                                           input_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
                    lstm.set_bias_for_gate(rela_index, gates[i], False,
                                           rec_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])

                rela_index = 2 * layerIndex + 1
                input_weights = weights[index:index + len(gates) * iw_count]
                index += iw_count * len(gates)
                rec_weights = weights[index:index + len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE]
                index += len(gates) * self.HIDDEN_SIZE * self.HIDDEN_SIZE
                input_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
                index += self.HIDDEN_SIZE * len(gates)
                rec_bias = weights[index:index + self.HIDDEN_SIZE * len(gates)]
                index += self.HIDDEN_SIZE * len(gates)
                hidden_2 = self.HIDDEN_SIZE ** 2

                for i in range(len(gates)):
                    lstm.set_weights_for_gate(rela_index, gates[i], True,
                                              input_weights[i * iw_count:i * iw_count + iw_count])
                    lstm.set_weights_for_gate(rela_index, gates[i], False,
                                              rec_weights[i * hidden_2:i * hidden_2 + hidden_2])
                    lstm.set_bias_for_gate(rela_index, gates[i], True,
                                           input_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
                    lstm.set_bias_for_gate(rela_index, gates[i], False,
                                           rec_bias[i * self.HIDDEN_SIZE:i * self.HIDDEN_SIZE + self.HIDDEN_SIZE])
            embedding_shape = (self.ALPHABET_SIZE, self.HIDDEN_SIZE*2)
            embedding_weights = weights[index:index + embedding_shape[0] * embedding_shape[1]]
            index += embedding_shape[0] * embedding_shape[1]

            embedding = network.add_constant(embedding_shape, embedding_weights)
            matrix_multiplication = network.add_matrix_multiply(lstm.get_output(0), trt.MatrixOperation.NONE,
                                                                embedding.get_output(0), trt.MatrixOperation.TRANSPOSE)

            embedding_bias_shape = (1, self.ALPHABET_SIZE)
            embedding_bias_weights = weights[index:index + embedding_bias_shape[1]]
            index += embedding_bias_shape[1]
            bias = network.add_constant(embedding_bias_shape, embedding_bias_weights)
            add_bias = network.add_elementwise(matrix_multiplication.get_output(0), bias.get_output(0),
                                               trt.ElementWiseOperation.SUM)

            dimensions = network.add_input('dimensions', trt.DataType.INT32, (self.SEQUENCE_SIZE, 1))
            softmax = network.add_ragged_softmax(add_bias.get_output(0), dimensions)

            builder.max_batch_size = self.max_batch_size
            builder_constants.max_workspace_size = 1 << 30

            network.mark_output(countryLayer.get_output(0))
            network.mark_output(softmax.get_output(0))
            engine = builder.build_engine(network, builder_constants)
            return engine


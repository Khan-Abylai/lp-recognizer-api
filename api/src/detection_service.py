import os

import numpy as np
import tensorrt as trt
import tqdm
from numba import cuda
from scipy.special import expit as sigmoid


class DetectionEngine(object):
    __MODELS_PATH = "./models/detection_weights.np"
    __ENGINE_NAME = 'detection.engine'

    def __init__(self, img_w=512, img_h=512, grid_size=16):
        self.IMG_SIZES = (3, 512, 512)
        self.coordinate_sizes = [13]
        self.bbox_attrs = 13
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.img_w, self.img_h = img_w, img_h
        self.grid_size = grid_size
        self.grid_w = self.img_w // grid_size
        self.grid_h = self.img_h // grid_size
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        if os.path.exists(DetectionEngine.__ENGINE_NAME):
            with open(DetectionEngine.__ENGINE_NAME, 'rb') as f, trt.Runtime(
                    self.trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            print("Model loaded")
        else:
            self.engine = self.__create_inference_engine(DetectionEngine.__MODELS_PATH)
            with open(DetectionEngine.__ENGINE_NAME, 'wb') as f:
                f.write(self.engine.serialize())
            print("Model built")
        self.x_y_offset = np.stack(np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h)), axis=2)
        self.execution_context = self.engine.create_execution_context()

    def predict(self, imgs):
        cuda_stream = cuda.stream()
        batch_size = 1
        plate_output = np.empty((self.bbox_attrs, self.grid_h, self.grid_w), dtype=np.float32)

        cuda_imgs = cuda.to_device(imgs, cuda_stream)
        cuda_plate_output = cuda.to_device(plate_output, cuda_stream)

        bindings = [cuda_imgs.device_ctypes_pointer.value,
                    cuda_plate_output.device_ctypes_pointer.value]

        self.execution_context.execute_async(batch_size, bindings, cuda_stream.handle.value, None)
        cuda_stream.synchronize()

        cuda_plate_output.copy_to_host(plate_output, stream=cuda_stream)

        plate_output = plate_output.reshape(batch_size, self.bbox_attrs, self.grid_h, self.grid_w)
        plate_output = np.transpose(plate_output, (0, 2, 3, 1))

        plate_output[..., :2] = sigmoid(plate_output[..., :2])
        plate_output[..., -1] = sigmoid(plate_output[..., -1])

        plate_output[..., 2:4] = np.exp(plate_output[..., 2:4])
        plate_output[..., :2] = plate_output[..., :2] + self.x_y_offset

        plate_output[..., :-1] = plate_output[..., :-1] * self.grid_size

        return plate_output.reshape(batch_size, self.grid_w * self.grid_h, self.bbox_attrs)

    def __create_inference_engine(self, weights_path):
        weights = np.fromfile(weights_path, dtype=np.float32)
        with trt.Builder(self.trt_logger) as builder, builder.create_network() as network:
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 30

            input_layer = network.add_input('data', trt.DataType.FLOAT, self.IMG_SIZES)
            kernel_size = (3, 3)
            stride = (1, 1)
            padding = (1, 1)
            channels = [16, 32, 64, 128, 256]

            index = 0

            prev_layer = input_layer
            features = []
            for i in tqdm.tqdm(range(len(channels))):
                for j in range(2):
                    conv_weights_count = prev_layer.shape[0] * channels[i] * kernel_size[0] * kernel_size[1]
                    conv_weights = weights[index:index + conv_weights_count]
                    index += conv_weights_count

                    conv_biases_count = channels[i]
                    conv_biases = weights[index:index + conv_biases_count]
                    index += conv_biases_count

                    conv_layer = network.add_convolution(prev_layer, channels[i], kernel_size, conv_weights,
                                                         conv_biases)
                    conv_layer.stride = stride
                    conv_layer.padding = padding

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

                if i == 4:
                    features.append(prev_layer)

                if i < 4:
                    pooling = network.add_pooling(prev_layer, trt.PoolingType.MAX, (2, 2))
                    pooling.stride = (2, 2)
                    prev_layer = pooling.get_output(0)
            for i, prev_layer in enumerate(features):
                conv_weights_count = prev_layer.shape[0] * self.coordinate_sizes[i]
                conv_weights = weights[index:index + conv_weights_count]
                index += conv_weights_count

                conv_biases_count = self.coordinate_sizes[i]
                conv_biases = weights[index:index + conv_biases_count]
                index += conv_biases_count

                conv_layer = network.add_convolution(prev_layer, self.coordinate_sizes[i], (1, 1), conv_weights,
                                                     conv_biases)

                network.mark_output(conv_layer.get_output(0))
            return builder.build_cuda_engine(network)

"""
Implementation of prediction on NVIDIA GPUs using TensorRT

Author: AdamP 2020-2020
"""

import os

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from Alpha4.model import DNNPredict


class DNNPredictRT(DNNPredict):
    """Implementation of prediction on NVIDIA GPUs using TensorRT"""
    def __init__(self, log, input_dim, output_dim, **kwargs):
        DNNPredict.__init__(self, log, input_dim, output_dim, **kwargs)
        self.tensorrt_predict = None

    def predict(self, state):
        """
        Prediction of value and policy based on an input state.
        Can be executed on freezed TensorRT models on GPU.
        :param state: Input tensor of a game state.
        :return: value: single value describing the chance to win
        :return: policy: tensor describing which next position should be investigated
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        prediction = self.tensorrt_predict(state)
        value = prediction[1].numpy()
        policy = prediction[0].numpy()
        return value[0, 0], policy

    def save(self, path):
        """Save model weights and convert to TensorRT"""
        DNNPredict.save(self, path)  # save model weights
        DNNPredictRT._convert(os.path.join(path, 'saved'), os.path.join(path, 'trt'))
        self.load(path)  # load converted model

    def load(self, path):
        """Load model weights and load converted TensorRT model"""
        DNNPredict.load(self, path)  # load model weights
        frozen_model = tf.saved_model.load(os.path.join(path, 'trt'), tags=[trt.tag_constants.SERVING])
        graph_func = frozen_model.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.tensorrt_predict = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

    @staticmethod
    def _convert(path_saved_model, path_frozen_model):
        """Convert to TensorRT model"""
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
        conversion_params = conversion_params._replace(precision_mode="FP16")
        conversion_params = conversion_params._replace(maximum_cached_engines=100)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=path_saved_model,
                                            conversion_params=conversion_params)
        converter.convert()
        converter.save(path_frozen_model)

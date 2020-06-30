import os
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from Alpha4.model import DNNPredict


class DNNPredictRT(DNNPredict):
    def __init__(self, input_dim, output_dim):
        DNNPredict.__init__(self, input_dim, output_dim)
        self.frozen_predict = None

    def predict(self, input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        prediction = self.frozen_predict(input)
        value = prediction[1].numpy()
        policy = prediction[0].numpy()
        return value[0, 0], policy

    def save(self, path):
        DNNPredict.save(self, path)
        DNNPredictRT._convert(os.path.join(path, 'saved'), os.path.join(path, 'frozen'))
        self.load(path)  # load converted model

    def load(self, path):
        DNNPredict.load(self, path)
        frozen_model = tf.saved_model.load(os.path.join(path, 'frozen'), tags=[trt.tag_constants.SERVING])
        graph_func = frozen_model.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.frozen_predict = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

    @staticmethod
    def _convert(path_saved_model, path_frozen_model):
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(max_workspace_size_bytes=(1 << 32))
        conversion_params = conversion_params._replace(precision_mode="FP16")
        conversion_params = conversion_params._replace(maximum_cached_engines=100)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=path_saved_model,
                                            conversion_params=conversion_params)
        converter.convert()
        converter.save(path_frozen_model)

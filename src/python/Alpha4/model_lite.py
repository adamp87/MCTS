import os
import numpy as np
import tensorflow as tf

from Alpha4.model import DNNPredict


class DNNPredictLite(DNNPredict):
    def __init__(self, input_dim, output_dim):
        DNNPredict.__init__(self, input_dim, output_dim)
        self.interpreter = None
        self.tflite_model = None
        self.input_details = None
        self.output_details = None

    def predict(self, state):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], state)

        self.interpreter.invoke()
        policy = self.interpreter.get_tensor(self.output_details[0]['index'])
        value = self.interpreter.get_tensor(self.output_details[1]['index'])

        return value[0, 0], policy

    def save(self, path):
        def representative_dataset_gen():
            num_calibration_steps = 128
            for i in range(num_calibration_steps):
                data = np.zeros((1, 6, 7, 9), dtype=np.float32)
                if i % 2 == 1:
                    data[0, :, :, 2] = 1
                x = np.random.randint(0, 6, 8)
                y = np.random.randint(0, 7, 8)
                data[0, x, y, 0] = 1
                x = np.random.randint(0, 6, 8)
                y = np.random.randint(0, 7, 8)
                data[0, x, y, 1] = 1
                yield [data]
        DNNPredict.save(self, path)
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path, 'saved'))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset_gen
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.int8
        # converter.experimental_new_converter = False
        self.tflite_model = converter.convert()

        # Save the TF Lite model
        with tf.io.gfile.GFile(os.path.join(path, 'tflite'), 'wb') as f:
            f.write(self.tflite_model)

        self.load(path)  # load converted model

    def load(self, path):
        DNNPredict.load(self, path)
        with tf.io.gfile.GFile(os.path.join(path, 'tflite'), 'rb') as f:
            self.tflite_model = f.read()

        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

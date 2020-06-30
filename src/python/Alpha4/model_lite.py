import os
import tensorflow as tf

from Alpha4.model import DNNPredict


class DNNPredictLite(DNNPredict):
    def __init__(self, input_dim, output_dim):
        DNNPredict.__init__(self, input_dim, output_dim)
        self.interpreter = None
        self.tflite_model = None
        self.input_details = None
        self.output_details = None

    def predict(self, input):
        input = tf.convert_to_tensor(input, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input)

        self.interpreter.invoke()
        policy = self.interpreter.get_tensor(self.output_details[0]['index'])
        value = self.interpreter.get_tensor(self.output_details[1]['index'])

        return value[0, 0], policy

    def save(self, path):
        DNNPredict.save(self, path)
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path, 'saved'))
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

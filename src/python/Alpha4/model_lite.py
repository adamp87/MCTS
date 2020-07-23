"""
Implementation of prediction with TensorLite, compatible with Edge TPU

Author: AdamP 2020-2020
"""

import os
import subprocess

import numpy as np
import tensorflow as tf

from Alpha4.model import DNNPredict


class DNNPredictLite(DNNPredict):
    """Implementation of prediction with TensorLite, compatible with Edge TPU"""
    def __init__(self, log, input_dim, output_dim, **kwargs):
        DNNPredict.__init__(self, log, input_dim, output_dim, **kwargs)
        self.interpreter = None
        self.tflite_model = None
        self.input_details = None
        self.output_details = None
        self.database = kwargs["database"]
        self.delegate = kwargs["delegate"]
        self.compile_tpu = kwargs["compile_tpu"]

    def predict(self, state):
        """
        Prediction of value and policy based on an input state.
        Can be executed on freezed TFLite models on CPU or TPU.
        :param state: Input tensor of a game state.
        :return: value: single value describing the chance to win
        :return: policy: tensor describing which next position should be investigated
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], state)

        self.interpreter.invoke()
        policy = self.interpreter.get_tensor(self.output_details[0]['index'])
        value = self.interpreter.get_tensor(self.output_details[1]['index'])

        return value[0, 0], policy

    def save(self, path):
        """Save model weights and convert to TFLite CPU and TPU"""
        def representative_dataset_gen():
            # generate (1, W, H, C) tensors
            for i in range(state.shape[0]):
                yield [state[i, :, :, :][np.newaxis, :, :, :]]

        # load states from database for generation of representative dataset
        if self.database.get_state_count() == 0:
            self.log.error('No state has been saved in database. Please generate states first with DNNPredict')
            exit(-1)
        state, _, _ = self.database.load(8192)

        # save model weights
        DNNPredict.save(self, path)

        # convert model to TF Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(path, 'saved'))
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset_gen
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.int8
        # converter.experimental_new_converter = False
        self.tflite_model = converter.convert()

        # save the TF Lite model
        with tf.io.gfile.GFile(os.path.join(path, 'tflite'), 'wb') as f:
            f.write(self.tflite_model)

        # compile TF Lite model to Edge TPU model
        if self.compile_tpu:
            cmd = ['edgetpu_compiler', os.path.join(path, 'tflite'), '-o', path]
            tpu_compile = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in tpu_compile.stdout:
                self.log.debug(line.decode().replace('\n', ''))
            tpu_compile.wait()
            if tpu_compile.returncode != 0:
                self.log.error("Failed to compile for TPU")
                exit(-1)
            self.log.info("Compiled TPU model and saved to {0}".format(os.path.join(path, 'tflite_edgetpu.tflite')))

        self.load(path)  # load converted model

    def load(self, path):
        """Load model weights and load TFLite CPU or TPU"""
        DNNPredict.load(self, path)  # load model weights

        # load TF Lite model, CPU or Edge TPU
        tf_name = 'tflite_edgetpu.tflite' if self.delegate is not None else 'tflite'
        with tf.io.gfile.GFile(os.path.join(path, tf_name), 'rb') as f:
            self.tflite_model = f.read()

        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model, experimental_delegates=self.delegate)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

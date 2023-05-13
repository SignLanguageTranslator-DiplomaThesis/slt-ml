import numpy as np
import tensorflow as tf

from constants import constants


class SignClassifier(object):

    """
    The __init__ method initializes the TensorFlow Lite interpreter with the pre-trained model and sets the input
    and output details. The model path and number of threads can be specified as arguments to the constructor.

    :param str model_path: Path of the pre-trained TensorFlow Lite model.
    :param int num_threads: Number of threads for the TensorFlow Lite interpreter.
    """
    def __init__(
        self,
        model_path=constants.TFLITE_SAVE_PATH,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    """
    It takes a landmark_list argument, which is a list of keypoint landmarks detected by MediaPipe. 
    These keypoints are passed as an input to the TensorFlow Lite interpreter, which produces an output 
    tensor that represents the predicted sign language gesture. The index of the highest value in the 
    output tensor is returned as the result.
    
    Here's a breakdown of the classify method:

    - input_details_tensor_index retrieves the index of the input tensor in the interpreter.
    - self.interpreter.set_tensor sets the input tensor to the landmark_list argument.
    - self.interpreter.invoke() runs the interpreter to produce an output tensor.
    - output_details_tensor_index retrieves the index of the output tensor in the interpreter.
    - self.interpreter.get_tensor retrieves the output tensor.
    - result_index finds the index of the highest value in the output tensor, which represents the predicted sign 
    language gesture.
    """
    def classify(self, landmark_list):
        input_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_tensor_index, np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_tensor_index = self.output_details[0]['index']
        output = self.interpreter.get_tensor(output_tensor_index)
        return np.argmax(np.squeeze(output))

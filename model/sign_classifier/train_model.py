import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflowjs

from constants import constants
from csv_parser.csv_parser import CsvParser


def load_dataset():
    """
    Load the dataset of hand landmark coordinates and labels from the CSV file.

    x_dataset - features - the normalized relative coordinates of the 21 hand landmarks

    y_dataset - labels = the label of the sign language gesture

    :return tuple: The created datasets.
    """

    x_dataset = np.loadtxt(constants.DATASET_PATH,
                           delimiter=',',
                           dtype='float32',
                           usecols=list(range(1, constants.NO_OF_LANDMARK_COORDINATES + 1)))

    y_dataset = np.loadtxt(constants.DATASET_PATH,
                           delimiter=',',
                           dtype='int32',
                           usecols=0)
    return x_dataset, y_dataset


class Model:
    def __init__(self):
        self.model = self.build()

    @staticmethod
    def build():
        """
        Defines the architecture (layers) of the deep neural network model and compiles it.

        :return tf.keras.models.Sequential: The created model.
        """

        # Define the architecture of the model
        model = tf.keras.models.Sequential(
            layers=[
                # input shape of the model - the number of landmark coordinates
                tf.keras.layers.Input((constants.NO_OF_LANDMARK_COORDINATES,), name="input_landmark_coordinates"),
                # randomly drops out some input units with a probability of 0.2 during training,
                # which helps to prevent over-fitting
                tf.keras.layers.Dropout(0.2, name="dropout_1"),
                # performs a linear transformation on the input data, followed by the ReLU activation function to
                # introduce non-linearity
                tf.keras.layers.Dense(128, activation='relu', use_bias=True, name="dense_1"),
                # normalizes the activations of the previous layer at each batch,
                # which helps improve training speed and generalization
                tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
                # randomly drops out some input units with a probability of 0.4 during training
                tf.keras.layers.Dropout(0.4, name="dropout_2"),
                tf.keras.layers.Dense(64, activation='relu', use_bias=True, name="dense_2"),
                tf.keras.layers.BatchNormalization(name="batch_normalization_2"),
                tf.keras.layers.Dense(32, activation='relu', use_bias=True, name="dense_3"),
                tf.keras.layers.BatchNormalization(name="batch_normalization_3"),
                tf.keras.layers.Dense(constants.NO_OF_CLASSES, activation='softmax', name="output_sign_label")
            ],
            name="sign_classifier"
        )

        # The model is compiled with:
        # -> Adam optimizer                                 = a stochastic gradient descent optimization algorithm;
        # -> sparse categorical cross-entropy loss function = used for multi-class classification problems.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def save_architecture(self):
        """
        Saves the summary of the architecture of the model and creates a plot of the neural network graph
        """

        # Print a summary of the model's architecture
        with open(constants.MODEL_SUMMARY_PATH, mode="w") as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))

        # Create a plot of the neural network graph, to visualise the model
        tf.keras.utils.plot_model(self.model,
                                  to_file=constants.NEURAL_NETWORK_VISUALIZATION_PATH,
                                  show_shapes=True,
                                  show_layer_names=True,
                                  show_layer_activations=True)

    def train(self, x_train, y_train, x_test, y_test):
        """
        Trains the model with the fit() function on the train data, with several callbacks,
        then evaluates it on the test data. Saves the model to a specified path.

        :param numpy.ndarray x_train: Training dataset.
        :param numpy.ndarray y_train: Set of labels for all the data in x_train.
        :param numpy.ndarray x_test: Test dataset.
        :param numpy.ndarray y_test: Set of labels for all the data in x_test.
        """

        # Used to save the model after each epoch if the validation loss has improved. The saved model can later be used
        # to make predictions on new data.
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            constants.MODEL_SAVE_PATH,
            verbose=True,
            save_weights_only=False,
            save_best_only=True,
        )

        # Used to stop the training process if the validation loss does not improve after 5 epochs.
        # This helps prevent over-fitting.
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=15,
            verbose=True
        )

        # Used to adjust the learning rate during training if the validation loss has stopped improving.
        # This can help the model converge to a better solution.
        lr_scheduler_callback = tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )

        self.model.fit(
            x_train,
            y_train,
            epochs=constants.NO_OF_EPOCHS,
            batch_size=constants.BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler_callback]
        )

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, batch_size=constants.BATCH_SIZE)

        with open(constants.CLASSIFICATION_REPORT_PATH, mode="w") as file:
            file.write('Validation loss: %.5f\n' % loss)
            file.write('Validation accuracy: %.5f\n' % accuracy)
            file.write("\n\n\n")

    def save(self):
        self.save_architecture()
        self.model.save(constants.MODEL_SAVE_PATH, include_optimizer=False)

    def save_as_tflite(self):
        """
        Convert and save the neural network model as TFLite.
        """

        # Creates a TFLite converter object and initializes it with the trained TensorFlow model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # Applies a set of default optimizations to improve the performance of the converted model
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized_model = converter.convert()

        with open(constants.TFLITE_SAVE_PATH, 'wb') as file:
            file.write(tflite_quantized_model)

    def save_as_js(self):
        tensorflowjs.converters.save_keras_model(self.model, constants.SAVED_MODELS_DIR)

    def predict_test_output(self, x_test):
        """
        Predicts the output of the test set using the trained model.

        :param numpy.ndarray x_test: Test data set.

        :return numpy.ndarray: Predicted class label.
        """

        # Use the predict method to obtain the predicted probabilities for each class for all samples in the test set
        y_pred_prob = self.model.predict(x_test)
        # Obtain the index of the class with the highest predicted probability for each sample, which is the
        # predicted class label
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred

    @staticmethod
    def generate_confusion_matrix(y_true, y_pred):
        """
        Generates the confusion matrix that is used to define the performance of the classification algorithm.

        :param numpy.ndarray y_true: True labels.
        :param numpy.ndarray y_pred: Predicted labels.
        """

        labels_indices = sorted(list(set(y_true)))
        labels = CsvParser().read_sign_labels()

        # Compute the confusion matrix using the true labels (y_true), predicted labels (y_pred),
        # and the list of label indices (labels_indices)
        confusion_matrix_data = confusion_matrix(y_true, y_pred, labels=labels_indices)

        # Create a Pandas DataFrame from the confusion matrix data.
        # It sets the row and column labels to the actual label names (labels).
        df_cmx = pd.DataFrame(confusion_matrix_data, index=labels, columns=labels)

        # Create a heatmap plot of the confusion matrix using seaborn and matplotlib
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df_cmx, annot=True, fmt='g', square=False, cmap='coolwarm')
        ax.set_ylim(len(set(y_true)), 0)
        plt.savefig(constants.CONFUSION_MATRIX_PATH)

    @staticmethod
    def generate_classification_report(y_test, y_pred):
        """
        Generate the Classification Report to a text file.

        :param numpy.ndarray y_test: Set of labels for all the data in x_test.
        :param numpy.ndarray y_pred: Predicted labels.
        """

        labels = CsvParser().read_sign_labels()
        with open(constants.CLASSIFICATION_REPORT_PATH, mode="a") as file:
            file.write("Classification Report\n\n\n")
            file.write(classification_report(y_test, y_pred, target_names=labels))


def classify_test_dataset_with_tflite(x_test):
    """
    Use a TFLite interpreter to classify the test dataset by performing inference on the TFLite model.

    :param numpy.ndarray x_test: Test dataset.
    """

    # Create a TFLite interpreter and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=constants.TFLITE_SAVE_PATH)
    interpreter.allocate_tensors()

    # Get input and output details from the interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with the first sample from the test dataset
    interpreter.set_tensor(input_details[0]['index'], np.array([x_test[0]]))

    # Invoke the interpreter to perform inference
    interpreter.invoke()

    # Get the output tensor from the interpreter
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    # Print the results
    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))


def main():
    # load the dataset
    x_dataset, y_dataset = load_dataset()

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_dataset,
                                                        y_dataset,
                                                        train_size=constants.TRAIN_SIZE,
                                                        random_state=constants.RANDOM_SEED)

    model = Model()
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    model.save()

    y_pred = model.predict_test_output(x_test)

    model.generate_confusion_matrix(y_test, y_pred)
    model.generate_classification_report(y_test, y_pred)

    model.save_as_tflite()
    model.save_as_js()

    classify_test_dataset_with_tflite(x_test)


if __name__ == "__main__":
    main()

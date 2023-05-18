import os
import cv2 as cv
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


def test_GPU_availability():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


def load_dataset(directory_path):
    x_dataset = []  # images
    y_dataset = []  # labels

    dir_list = os.listdir(directory_path)
    for i in range(len(dir_list)):
        print("Obtaining images of", dir_list[i], "...")
        for image in os.listdir(os.path.join(directory_path, dir_list[i])):
            img = cv.imread(os.path.join(directory_path, dir_list[i], image))
            img = cv.resize(img, (32, 32))
            x_dataset.append(img)
            y_dataset.append(i)

    return x_dataset, y_dataset


def plot_sample_images():
    figure = plt.figure()
    plt.figure(figsize=(16, 5))

    for i in range(0, 26):
        plt.subplot(3, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        path = os.path.join(
            constants.TRAIN_DATASET_PATH,
            f"{constants.IMAGE_CLASSES[i]}",
            f"{constants.IMAGE_CLASSES[i]}1.jpg"
        )
        img = plt.imread(path)
        plt.imshow(img)
        plt.xlabel(constants.IMAGE_CLASSES[i])


def preprocess_data(x_dataset, y_dataset):
    np_x = np.array(x_dataset)
    normalised_x = np_x.astype('float32') / 255.0

    label_encoded_y = tf.keras.utils.to_categorical(y_dataset)

    x_train, x_test, y_train, y_test = train_test_split(
        normalised_x,
        label_encoded_y,
        train_size=constants.TRAIN_SIZE,
        random_state=constants.RANDOM_SEED
    )

    print("Training data: ", x_train.shape)
    print("Test data: ", x_test.shape)

    return x_train, x_test, y_train, y_test


class Model:
    def __init__(self):
        self.history = None
        self.model = self.build()

    @staticmethod
    def build():
        """
        Defines the architecture (layers) of the deep neural network model and compiles it.

        :return tf.keras.models.Sequential: The created model.
        """
        model = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Conv2D(256, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(constants.IMAGE_CLASSES, activation='softmax'),
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
        Saves the summary of the architecture of the model and creates a plot of the neural network graph.
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

        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=constants.NO_OF_EPOCHS,
            batch_size=constants.BATCH_SIZE,
            validation_data=(x_test, y_test),
            shuffle=True,
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

    def plot_results(self):
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 2, 1)
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()

    @staticmethod
    def generate_confusion_matrix(y_true, y_pred):
        """
        Generates the confusion matrix that is used to define the performance of the classification algorithm.

        :param numpy.ndarray y_true: True labels.
        :param numpy.ndarray y_pred: Predicted labels.
        """

        labels_indices = sorted(list(set(y_true)))

        # Compute the confusion matrix using the true labels (y_true), predicted labels (y_pred),
        # and the list of label indices (labels_indices)
        confusion_matrix_data = confusion_matrix(y_true, y_pred, labels=labels_indices)

        # Create a Pandas DataFrame from the confusion matrix data.
        # It sets the row and column labels to the actual label names (labels).
        df_cmx = pd.DataFrame(confusion_matrix_data, index=constants.IMAGE_CLASSES, columns=constants.IMAGE_CLASSES)

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

        with open(constants.CLASSIFICATION_REPORT_PATH, mode="a") as file:
            file.write("Classification Report\n\n\n")
            file.write(classification_report(y_test, y_pred, target_names=constants.IMAGE_CLASSES))


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
    x_dataset, y_dataset = load_dataset(constants.TRAIN_DATASET_PATH)
    plot_sample_images()

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = preprocess_data(x_dataset, y_dataset)

    model = Model()
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    model.save()

    model.plot_results()

    y_pred = model.predict_test_output(x_test)

    model.generate_confusion_matrix(y_test, y_pred)
    model.generate_classification_report(y_test, y_pred)

    model.save_as_tflite()
    model.save_as_js()

    classify_test_dataset_with_tflite(x_test)


if __name__ == "__main__":
    main()

import os.path
import sys

directory = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(directory)

import constants
from csv_parser.csv_parser import CsvParser

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# Load the dataset of hand landmarks and labels from the CSV file
def load_dataset():
    # features = the normalized relative coordinates of the 21 hand landmarks
    x_dataset = np.loadtxt(constants.DATASET_PATH,
                           delimiter=',',
                           dtype='float32',
                           usecols=list(range(1, constants.NO_OF_LANDMARK_COORDINATES + 1)))
    # labels = the label of the sign language gesture
    y_dataset = np.loadtxt(constants.DATASET_PATH,
                           delimiter=',',
                           dtype='int32',
                           usecols=0)
    return x_dataset, y_dataset


# Defines the layers of the deep neural network model and compiles the model
def create_model():
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

    # Print a summary of the model's architecture
    with open(constants.MODEL_SUMMARY_PATH, mode="w") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Create a plot of the neural network graph, to visualise the model
    tf.keras.utils.plot_model(model,
                              to_file=constants.NEURAL_NETWORK_VISUALIZATION_PATH,
                              show_shapes=True,
                              show_layer_names=True,
                              show_layer_activations=True)

    # The model is compiled with:
    # -> Adam optimizer                                 = a stochastic gradient descent optimization algorithm;
    # -> sparse categorical cross-entropy loss function = used for multi-class classification problems.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
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

    model.fit(
        x_train,
        y_train,
        epochs=constants.NO_OF_EPOCHS,
        batch_size=constants.BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler_callback]
    )

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=constants.BATCH_SIZE)

    with open(constants.CLASSIFICATION_REPORT_PATH, mode="w") as file:
        file.write('Validation loss: %.5f\n' % loss)
        file.write('Validation accuracy: %.5f\n' % accuracy)
        file.write("\n\n\n")

    model.save(constants.MODEL_SAVE_PATH, include_optimizer=False)


# Predicts the output of the test set using the trained model
def predict_test_output(model, x_test):
    # Use the predict method to obtain the predicted probabilities for each class for all samples in the test set
    y_pred_prob = model.predict(x_test)
    # Obtain the index of the class with the highest predicted probability for each sample, which is the predicted class
    # label
    y_pred = np.argmax(y_pred_prob, axis=1)
    return y_pred


# Function to print confusion matrix and classification report
def save_confusion_matrix(y_true, y_pred, y_test):
    labels_indices = sorted(list(set(y_true)))
    labels = CsvParser().read_sign_labels()

    # Compute the confusion matrix using the true labels (y_true), predicted labels (y_pred),
    # and the list of label indices (labels_indices)
    confusion_matrix_data = confusion_matrix(y_true, y_pred, labels=labels_indices)

    # This line creates a Pandas DataFrame from the confusion matrix data.
    # It sets the row and column labels to the actual label names (labels).
    df_cmx = pd.DataFrame(confusion_matrix_data, index=labels, columns=labels)

    # Create a heatmap plot of the confusion matrix using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False, cmap='coolwarm')
    ax.set_ylim(len(set(y_true)), 0)
    plt.savefig(constants.CONFUSION_MATRIX_PATH)

    # Generate the Classification Report to a text file
    with open(constants.CLASSIFICATION_REPORT_PATH, mode="a") as file:
        file.write("Classification Report\n\n\n")
        file.write(classification_report(y_test, y_pred))


# Function to convert and save the model as TFLite
def save_as_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(constants.TFLITE_SAVE_PATH, 'wb').write(tflite_quantized_model)


def classify_test_dataset_with_tflite(x_test):
    interpreter = tf.lite.Interpreter(model_path=constants.TFLITE_SAVE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.array([x_test[0]]))

    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

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

    model = create_model()

    train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    y_pred = predict_test_output(model, x_test)

    save_confusion_matrix(y_test, y_pred, y_test)

    save_as_tflite(model)

    classify_test_dataset_with_tflite(x_test)


if __name__ == "__main__":
    main()

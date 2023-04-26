import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from constants import RANDOM_SEED, DATASET_PATH, MODEL_SAVE_PATH, TFLITE_SAVE_PATH, NUM_CLASSES


# Load the dataset of hand landmarks and labels from the CSV file
def load_dataset():
    x_dataset = np.loadtxt(DATASET_PATH, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(DATASET_PATH, delimiter=',', dtype='int32', usecols=0)
    return x_dataset, y_dataset


def create_model():
    # Define the architecture of the ML model
    # The model is made up of several layers, including input, dropout, dense, and output layers

    # -> input layer           = has 42 nodes, corresponding to the (x, y) coordinates of the 21 landmarks of the hand,
    #                           each represented by a float value
    # -> dropout layers        = help prevent overfitting by randomly dropping out nodes during training
    # -> dense layers          = perform calculations on the input data and output their results to the next layer
    # -> final output layer    = has NUM_CLASSES nodes, which represent all the possible sign language gestures
    #                           that the model can classify
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Print a summary of the model's architecture
    # The summary provides information about:
    #   -> the number of layers
    #   -> the number of parameters in each layer
    #   -> the output shape of each layer
    model.summary()

    # The model is compiled with:
    # -> Adam optimizer                                 = a stochastic gradient descent optimization algorithm
    #                                                       that is commonly used in ML;
    # -> sparse categorical cross-entropy loss function = used for multi-class classification problems.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, verbose=1, save_weights_only=False)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)
    print('Validation loss:', val_loss)
    print('Validation accuracy:', val_acc)
    return val_loss, val_acc


# Function to print confusion matrix and classification report
def print_confusion_matrix(y_true, y_pred, y_test, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


# Function to convert and save the model as TFLite
def save_as_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(TFLITE_SAVE_PATH, 'wb').write(tflite_quantized_model)


def classify_landmarks_with_tflite(x_test):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
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
    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

    model = create_model()

    train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    print_confusion_matrix(y_test, y_pred, y_test)

    model.save(MODEL_SAVE_PATH, include_optimizer=False)

    save_as_tflite(model)

    classify_landmarks_with_tflite(x_test)


if __name__ == "__main__":
    main()


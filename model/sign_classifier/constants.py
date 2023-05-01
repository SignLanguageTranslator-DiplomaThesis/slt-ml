SIGN_LABELS_PATH = "model/sign_classifier/sign_classifier_label.csv"
WEBCAM_DEVICE_INPUT = 0

# Dimension of video capture
CAPTURE_FRAME_WIDTH = 960
CAPTURE_FRAME_HEIGHT = 540

# Set a random seed for reproducibility
RANDOM_SEED = 42

# Set the path to the dataset, model and tflite files
DATASET_PATH = 'model/sign_classifier/sign_dataset.csv'
MODEL_SAVE_PATH = 'model/sign_classifier/sign_classifier.hdf5'
TFLITE_SAVE_PATH = 'model/sign_classifier/sign_classifier.tflite'

# Paths for ML model information
MODEL_SUMMARY_PATH = 'model/sign_classifier/info/model_summary.txt'
CONFUSION_MATRIX_PATH = 'model/sign_classifier/info/confusion_matrix.png'
CLASSIFICATION_REPORT_PATH = 'model/sign_classifier/info/classification_report.txt'
NEURAL_NETWORK_VISUALIZATION_PATH = 'model/sign_classifier/info/sign_classifier_model.png'

# Define the number of classes (labels) in the dataset
NO_OF_CLASSES = 6

NO_OF_LANDMARKS = 21
NO_OF_LANDMARK_COORDINATES = 42
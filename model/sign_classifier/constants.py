SIGN_LABELS_PATH = "model/sign_classifier/sign_classifier_label.csv"
WEBCAM_DEVICE_INPUT = 0

# MediaPipe Hands configuration parameters
MAX_NUMBER_OF_HANDS = 2
USE_STATIC_IMAGE_MODE = False
MIN_DETECTION_CONFIDENCE = 0.75
MIN_TRACKING_CONFIDENCE = 0.6

# Dimension of video capture
CAPTURE_FRAME_WIDTH = 960
CAPTURE_FRAME_HEIGHT = 540

# Model configuration constants

# Set a random seed for reproducibility
RANDOM_SEED = 42
# The size of the training dataset, as a percentage from the total dataset
TRAIN_SIZE = 0.8
NO_OF_EPOCHS = 250
BATCH_SIZE = 128

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

# Desktop application modes
SAVE_SNAPSHOT_MODE = 115    # S
CREATE_LABEL_MODE = 108     # L
CHOOSE_LABEL_MODE = 99      # C
NORMAL_MODE = 110           # N
NOT_SELECTED = -1

modes = [SAVE_SNAPSHOT_MODE, CREATE_LABEL_MODE, CHOOSE_LABEL_MODE, NORMAL_MODE]

ESC_KEY = 27

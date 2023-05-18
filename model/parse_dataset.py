import os
import cv2 as cv
import mediapipe as mp
import argparse

from constants import constants
from csv_parser.csv_parser import CsvParser
from coordinates.coordinates import CoordinateConverter

JPG = '.jpg'
JPEG = '.jpeg'
PNG = '.png'
HORIZONTAL_AXIS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, help='Path to the folder containing the pictures')
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a valid folder path.")
        exit()

    return args


def process_image(hands, image, sign_index):
    results = hands.process(image)

    # Check if any hands are detected in the image
    if results.multi_hand_landmarks:
        # Iterate over each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Iterate over each landmark point
            coordinate_converter = CoordinateConverter(hand_landmarks)
            coordinate_converter.convert_to_pixel(image)
            coordinate_converter.convert_to_relative_and_normalize()

            # Write to the dataset file
            CsvParser.logging_csv(
                constants.SAVE_SNAPSHOT_MODE,
                sign_index,
                coordinate_converter.normalized_landmarks
            )


def parse_letter_directory(path, sign_index):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=constants.MAX_NUMBER_OF_HANDS,
        min_detection_confidence=constants.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=constants.MIN_TRACKING_CONFIDENCE,
    )

    for filename in os.listdir(path):
        if filename.endswith(JPEG) or filename.endswith(JPG) or filename.endswith(PNG):
            image_path = os.path.join(path, filename)
            image = cv.imread(image_path)

            # Convert the image to RGB for Mediapipe Hands
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Flip the image, to save the coordinates of the gesture for both hands - left & right
            # flipped_image = cv.flip(image_rgb, HORIZONTAL_AXIS)

            process_image(hands, image_rgb, sign_index)
            # process_image(hands, flipped_image, sign_index)

        # Release resources
        cv.destroyAllWindows()


def parse_all_letters():
    args = parse_args()

    # Read labels from sign_label CSV file
    sign_language_labels = CsvParser.read_sign_labels()

    for directory in os.listdir(args.folder_path):
        path = os.path.join(args.folder_path, directory)
        if os.path.isdir(path):
            letter = os.path.basename(path)
            if len(letter) == 1 and letter.isalpha():
                letter = letter.upper()
                sign_index = sign_language_labels.index(letter)
                print(letter)
                parse_letter_directory(path, sign_index)


if __name__ == "__main__":
    # parse_letter_directory(r"C:\Users\denis\OneDrive\Documents\UTCN\licenta\dataset\asl_alphabet_train\asl_alphabet_train\Q", 16)
    parse_all_letters()

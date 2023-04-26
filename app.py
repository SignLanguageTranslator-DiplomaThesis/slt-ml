import copy
import argparse

import cv2 as cv
import mediapipe as mp

import constants
from model import SignClassifier
from csv_parser.csv_parser import CsvParser
from interface.interface import UserInterface
from interface.annotations import Annotations


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing
    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation
    cap = cv.VideoCapture(constants.WEBCAM_DEVICE_INPUT)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, constants.CAPTURE_FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, constants.CAPTURE_FRAME_HEIGHT)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    sign_classifier = SignClassifier()

    # Read labels from sign_classifier_label CSV file
    sign_language_labels = CsvParser.read_sign_labels()

    number = -1

    while cap.isOpened():
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, number, sign_language_labels)

        # Camera capture
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = Annotations.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                processed_landmark_list = process_landmark(landmark_list)

                # Write to the dataset file
                CsvParser.logging_csv(mode, number, processed_landmark_list)

                # Hand sign classification
                hand_sign_id = sign_classifier.classify(processed_landmark_list)

                # DF: draws the 21 landmark points generated by MediaPipe and the lines connecting them
                debug_image = Annotations.draw_landmarks(debug_image, results.multi_hand_landmarks)

                debug_image = Annotations.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    sign_language_labels[hand_sign_id]
                )

        debug_image = Annotations.draw_info(debug_image, mode)

        # Screen reflection
        cv.imshow('Sign Language Translator', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, number, sign_language_labels):
    user_interface = UserInterface()
    mode = -1
    if key == 110:  # n - normal mode
        mode = 0
    elif key == 107:  # k - save new data in the dataset
        mode = 1
    elif key == 115:  # s - choose sign gesture (label) to perform
        user_interface.generate_label_dropdown(sign_language_labels)
        number = user_interface.selected_label_index
        mode = 2
        print(number)
    elif key == 108:  # l - save new label in the CSV file
        user_interface.generate_input_field(sign_language_labels)
        mode = 3
    return number, mode


def calc_landmark_list(image, landmarks):
    image_height, image_width, _ = image.shape

    landmark_list = []

    # Iterate through each landmark point
    for landmark in landmarks.landmark:
        # Convert landmark coordinates from normalized [0, 1] to pixel coordinates
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)

        # Append the pixel coordinates to the landmark list
        landmark_list.append([landmark_x, landmark_y])

    return landmark_list


def process_landmark(landmark_list):
    processed_landmark_list = []

    # Convert to relative coordinates
    # We use the first landmark point as the base point and subtract its x and y values from all other points
    base_x, base_y = landmark_list[0][:2]  # get the x and y values of the base point
    for landmark_point in landmark_list:
        x, y = landmark_point  # unpack the x and y values of the landmark point
        processed_landmark_list.extend([x - base_x, y - base_y])  # add the normalized x and y values to the output list

    # Normalization
    # We divide each value in the list by the maximum absolute value
    max_value = max(map(abs, processed_landmark_list))  # find the maximum absolute value in the list
    # Divide each value by the maximum absolute value
    processed_landmark_list = [landmark / max_value for landmark in processed_landmark_list]

    return processed_landmark_list


if __name__ == '__main__':
    main()

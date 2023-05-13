import cv2 as cv
import numpy as np
import mediapipe as mp

from constants import constants


class Annotations:
    @staticmethod
    def compute_bounding_box(landmarks):
        x, y, width, height = cv.boundingRect(np.array(landmarks))

        return [x, y, x + width, y + height]

    @staticmethod
    def draw_landmarks(image, multi_hand_landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return image

    @staticmethod
    def draw_info_text(image, bounding_box, handedness, hand_sign_text):
        cv.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ': ' + hand_sign_text
        cv.putText(image,
                   info_text,
                   (bounding_box[0] + 5, bounding_box[1] - 4),
                   cv.FONT_HERSHEY_DUPLEX,
                   0.6,
                   (255, 255, 255),
                   1,
                   cv.LINE_AA)

        return image

    @staticmethod
    def draw_info(image, mode):
        mode_string = {
            constants.SAVE_SNAPSHOT_MODE: "Save Snapshot",
            constants.CREATE_LABEL_MODE: "Create New Sign Label",
            constants.CHOOSE_LABEL_MODE: "Choose Sign to Perform",
            constants.NORMAL_MODE: "Normal Mode"
        }
        if mode != constants.NOT_SELECTED:
            cv.putText(image,
                       "MODE:" + mode_string[mode],
                       (10, 90),
                       cv.FONT_HERSHEY_DUPLEX,
                       0.6,
                       (255, 255, 255),
                       1,
                       cv.LINE_AA)
        return image

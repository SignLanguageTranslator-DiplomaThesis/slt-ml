import cv2 as cv
import numpy as np
import mediapipe as mp

from model.sign_classifier import constants


class Annotations:
    @staticmethod
    def calc_bounding_rect(image, landmarks):
        image_height, image_width = image.shape[0], image.shape[1]

        landmark_points = []
        for landmark in landmarks.landmark:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_points.append((landmark_x, landmark_y))

        x, y, w, h = cv.boundingRect(np.array(landmark_points))

        return [x, y, x + w, y + h]

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
    def draw_info_text(image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        return image

    @staticmethod
    def draw_info(image, mode):
        mode_string = {
            constants.SAVE_SNAPSHOT_MODE: "Save Snapshot",
            constants.CREATE_LABEL_MODE: "Create New Sign Label",
            constants.CHOOSE_LABEL_MODE: "Choose Sign to Perform",
            constants.NORMAL_MODE: "Normal Mode"
        }
        if mode > 0:
            cv.putText(image,
                       "MODE:" + mode_string[mode],
                       (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),
                       1,
                       cv.LINE_AA)
        return image

import csv
from constants import constants


class CsvParser:

    def __init__(self):
        pass

    # DF: Appends a new label inserted by the user to the [initial]_sign_label.csv file
    @staticmethod
    def write_label_to_csv(data, sign_language_labels):
        sign_language_labels.extend(data)
        with open(constants.SIGN_LABELS_PATH, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    @staticmethod
    def logging_csv(mode, number, landmark_list):
        if mode == constants.SAVE_SNAPSHOT_MODE and number != -1:
            with open(constants.TEST_DATASET_PATH, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        return

    @staticmethod
    def read_sign_labels():
        with open(constants.SIGN_LABELS_PATH, encoding='utf-8-sig') as f:
            sign_language_labels = csv.reader(f)
            return [
                row[0] for row in sign_language_labels
            ]

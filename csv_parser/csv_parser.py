import csv
import constants


class CsvParser:

    def __init__(self):
        pass

    # DF: Appends a new label inserted by the user to the sign_classifier_label.csv file
    @staticmethod
    def write_label_to_csv(data, sign_language_labels):
        sign_language_labels.extend(data)
        with open(constants.SIGN_LABELS_PATH, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    @staticmethod
    def logging_csv(mode, number, landmark_list):
        if mode == 1 and number != -1:
            csv_path = 'model/sign_classifier/sign_dataset.csv'
            with open(csv_path, 'a', newline="") as f:
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

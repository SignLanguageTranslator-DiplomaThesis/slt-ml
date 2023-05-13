class CoordinateConverter:
    def __init__(self, landmarks):
        self.initial_landmarks = landmarks
        self.pixel_landmarks = []
        self.normalized_landmarks = []

    def convert_to_pixel(self, image):
        image_height, image_width, _ = image.shape

        # Iterate through each landmark point
        for landmark in self.initial_landmarks.landmark:
            # Convert landmark coordinates from normalized [0, 1] to pixel coordinates
            landmark_x = int(landmark.x * image_width)
            landmark_y = int(landmark.y * image_height)

            # Append the pixel coordinates to the landmark list
            self.pixel_landmarks.append([landmark_x, landmark_y])

    def convert_to_relative_and_normalize(self):
        min_landmark_x = min(self.pixel_landmarks, key=lambda landmark: landmark[0])[0]
        min_landmark_y = min(self.pixel_landmarks, key=lambda landmark: landmark[1])[1]

        max_landmark_x = max(self.pixel_landmarks, key=lambda landmark: landmark[0])[0]
        max_landmark_y = max(self.pixel_landmarks, key=lambda landmark: landmark[1])[1]

        # Calculate the range of x and y values
        x_range = max_landmark_x - min_landmark_x
        y_range = max_landmark_y - min_landmark_y

        # Convert to relative coordinates and normalize
        for landmark_point in self.pixel_landmarks:
            x, y = landmark_point
            relative_x = (x - min_landmark_x) / x_range
            relative_y = (y - min_landmark_y) / y_range
            self.normalized_landmarks.extend([relative_x, relative_y])

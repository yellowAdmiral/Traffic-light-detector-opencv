import cv2
import numpy as np
import random
from rasrobot import RASRobot

class TrafficLightDetector:
    def __init__(self, debug=False):
        """
        Initialize the traffic light detector.

        Args:
        debug (bool): Flag to enable or disable debug mode.
        """
        self.debug = debug
        self.debug_seq = 0  # Sequence number for debugging

    def detect(self, image):
        """
        Detect the color of the traffic light in the provided image.

        This function processes the input image to identify the presence of traffic lights 
        and determine their color. It uses HSV color space to create masks for red, yellow, 
        and green colors, then applies morphological operations and contour detection to 
        find traffic light signals.

        Args:
        image (numpy.ndarray): The input image from the robot's camera.

        Returns:
        str: The color of the detected traffic light ('red', 'yellow', 'green', or 'none').
        """
        if image is None:
            return 'none'

        self.debug_seq += 1  # Increment debug sequence number

        # Convert the image to BGR format if it has an alpha channel.
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert the BGR image to HSV color space for better color segmentation.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for detecting red, yellow, and green colors.
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        green_lower = np.array([35, 100, 100])
        green_upper = np.array([90, 255, 255])

        # Create binary masks for the colors of interest.
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        # Apply morphological operations to clean up the masks.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        # Apply Gaussian blur to the masks to reduce noise.
        blurred_red = cv2.GaussianBlur(mask_red, (9, 9), 2)
        blurred_yellow = cv2.GaussianBlur(mask_yellow, (9, 9), 2)
        blurred_green = cv2.GaussianBlur(mask_green, (9, 9), 2)

        # Helper function to find and verify contours in the blurred mask.
        def find_contours(mask, color):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 20:  # Check if the detected area is large enough to be a traffic light.
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 <= aspect_ratio <= 1.2:  # Check if the detected shape is approximately square.
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if self.debug:
                            print(f"{self.debug_seq}: {color.capitalize()} light detected with area {area} at ({x}, {y}, {w}, {h}).")
                        return True
            return False

        # Define the region of interest (ROI) in the image to focus on the area likely containing traffic lights.
        height, width = image.shape[:2]
        roi_height_start = 0
        roi_height_end = height // 2
        roi_width_start = width // 2
        roi_width_end = width

        roi = image[roi_height_start:roi_height_end, roi_width_start:roi_width_end]

        # Resize the ROI to make distant objects more prominent.
        scale_percent = 200  # Resize factor
        width = int(roi.shape[1] * scale_percent / 100)
        height = int(roi.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_roi = cv2.resize(roi, dim, interpolation=cv2.INTER_LINEAR)

        blurred_red_roi = cv2.resize(blurred_red[roi_height_start:roi_height_end, roi_width_start:roi_width_end], dim, interpolation=cv2.INTER_LINEAR)
        blurred_yellow_roi = cv2.resize(blurred_yellow[roi_height_start:roi_height_end, roi_width_start:roi_width_end], dim, interpolation=cv2.INTER_LINEAR)
        blurred_green_roi = cv2.resize(blurred_green[roi_height_start:roi_height_end, roi_width_start:roi_width_end], dim, interpolation=cv2.INTER_LINEAR)

        # Display the HSV image for debugging purposes.
        if self.debug:
            cv2.imshow('HSV Image', hsv)
            cv2.waitKey(1)

        # Check for traffic light colors in the ROI.
        if find_contours(blurred_red_roi, 'red'):
            return "red"
        elif find_contours(blurred_yellow_roi, 'yellow'):
            return "yellow"
        elif find_contours(blurred_green_roi, 'green'):
            return "green"
        else:
            return "none"

class LaneController:
    def __init__(self, debug=False):
        """
        Initialize the lane controller with default parameters.

        Args:
        debug (bool): Flag to enable or disable debug mode.
        """
        self.mode = 'driving'  # Initial mode is set to driving
        self.turn_counter = 0  # Counter to manage turning behavior

        # Parameters for visualization
        self.steering_row = 94  # Row for visualizing steering
        self.crossroad_row = 90  # Row for visualizing crossroads

        # Control parameters
        self.max_steering_angle = 0.3  # Maximum allowable steering angle
        self.max_speed = 50  # Maximum speed (in units, e.g., cm/s)
        self.min_speed = 35  # Minimum speed (in units, e.g., cm/s)

        self.debug = debug  # Debug flag to enable visualization and logging

    def _display_image(self, image, name, scale=2):
        """
        Display the image if debug mode is enabled.

        Args:
        image (numpy.ndarray): The image to display.
        name (str): The name of the display window.
        scale (int): The scaling factor for the display window size.
        """
        if not self.debug:
            return

        # Highlight specific rows for visual reference
        image = image.copy()
        image[self.steering_row, :] = 1.0 - image[self.steering_row, :]
        image[self.crossroad_row, :] = 1.0 - image[self.crossroad_row, :]

        # Display the image in a window
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, image.shape[1] * scale, image.shape[0] * scale)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def _get_road(self, image):
        """
        Detect road markings using color segmentation to identify tarmac and yellow lines.

        Args:
        image (numpy.ndarray): The input image from the robot's camera.

        Returns:
        numpy.ndarray: A binary image representing the detected road markings.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for tarmac and yellow road markings
        lower_tarmac = np.array([0, 0, 0], np.uint8)
        upper_tarmac = np.array([200, 50, 100], np.uint8)
        mask_tarmac = cv2.inRange(hsv, lower_tarmac, upper_tarmac).astype(bool)

        lower_yellow = np.array([25, 100, 100], np.uint8)
        upper_yellow = np.array([50, 255, 255], np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow).astype(bool)

        # Create a binary image of the road markings
        binary_image = np.zeros(image.shape)
        binary_image[mask_tarmac] = 1.0
        binary_image[mask_yellow] = 1.0
        binary_image = binary_image[:, :, 0]  # Convert to grayscale

        # Reduce noise in the binary image using morphological operations
        reduce_noise = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, (3, 3))

        # Display the processed road image for debugging
        self._display_image(reduce_noise, 'road image')

        return reduce_noise

    def _get_road_center(self, road_image):
        """
        Calculate the center of the road based on the road image.

        This function analyzes the binary road image to find the center line of the road,
        which helps in determining the steering direction.

        Args:
        road_image (numpy.ndarray): The binary image representing the road.

        Returns:
        float: The normalized center position of the road (range [-1, 1]).
        """
        road_img_line = road_image[self.steering_row, :]
        if road_img_line.sum() == 0:
            return None

        # Calculate the center index and normalize it
        idx = road_img_line.nonzero()[0]
        cx = np.mean(idx / len(road_img_line))
        center_normalised = (cx - 0.5) * 2  # Normalize to range [-1, 1]

        return center_normalised

    def _at_crossroads(self, road_image):
        """
        Check if the robot is at a crossroads based on the analysis of the road image.

        This function compares the road width at the steering row and crossroad row to detect
        significant changes that indicate the presence of a crossroad.

        Args:
        road_image (numpy.ndarray): The binary image representing the road.

        Returns:
        bool: True if at a crossroads, False otherwise.
        """
        steering_sum = road_image[self.steering_row, :].sum()
        crossroad_sum = road_image[self.crossroad_row, :].sum()
        diff = crossroad_sum - steering_sum

        # Simple detection logic: if there's a significant road width difference
        if diff > 50 and crossroad_sum > 200:
            return True
        return False

    def _get_steering_angle(self, road_center):
        """
        Calculate the steering angle based on the detected center of the road.

        Args:
        road_center (float): The normalized center position of the road.

        Returns:
        float: The calculated steering angle.
        """
        if road_center is None:
            return 0

        # Calculate the desired steering angle to keep the robot centered on the road
        preferred_road_center_pos = -0.15
        angle = road_center - preferred_road_center_pos
        angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)

        return angle

    def _get_speed(self, steering_angle):
        """
        Calculate the speed based on the current steering angle.

        The speed is adjusted based on the steering angle to ensure stable and safe driving.

        Args:
        steering_angle (float): The current steering angle.

        Returns:
        float: The calculated speed.
        """
        speed_factor = (self.max_steering_angle - abs(steering_angle)) / self.max_steering_angle
        speed = (self.max_speed - self.min_speed) * speed_factor + self.min_speed

        return speed

    def get_angle_and_speed(self, image):
        """
        Main function to get the steering angle and speed based on the current driving mode.

        This function determines the robot's steering angle and speed by analyzing the road image,
        detecting the road center, and adjusting the speed accordingly.

        Args:
        image (numpy.ndarray): The input image from the robot's camera.

        Returns:
        tuple: The calculated steering angle and speed.
        """
        steering_angle = 0

        if self.mode == 'turning_left':
            # Mode for turning left at crossroads
            self.turn_counter += 1
            if self.turn_counter < 50:  # Go straight for a short duration to stabilize the turn
                steering_angle = 0
            elif self.turn_counter < 130:  # Turn left smoothly
                steering_angle = -self.max_steering_angle
            else:
                self.mode = 'driving'  # Return to normal driving mode
                self.turn_counter = 0

        elif self.mode == 'turning_right':
            # Mode for turning right at crossroads
            self.turn_counter += 1
            if self.turn_counter < 50:  # Go straight for a short duration to stabilize the turn
                steering_angle = 0
            elif self.turn_counter < 130:  # Turn right smoothly
                steering_angle = self.max_steering_angle
            else:
                self.mode = 'driving'  # Return to normal driving mode
                self.turn_counter = 0

        elif self.mode == 'driving':
            # Normal driving mode, follow the road
            road_image = self._get_road(image)
            road_center = self._get_road_center(road_image)

            if road_center is None:
                steering_angle = 0  # If no road center detected, maintain the current angle
            else:
                steering_angle = self._get_steering_angle(road_center)

            # Check if at crossroads and decide randomly to turn left or right
            if self._at_crossroads(road_image):
                self.mode = np.random.choice(['turning_left', 'turning_right'])  # Randomly select turn direction
                if self.debug:
                    print(f'Crossroads detected! Changing mode to {self.mode}')

        # Calculate speed based on the steering angle
        speed = self._get_speed(steering_angle)

        return steering_angle, speed

class MyRobot(RASRobot):
    def __init__(self):
        """
        Initialize the MyRobot class, including the traffic light detector and lane controller.
        """
        super(MyRobot, self).__init__()
        self.traffic_light_detector = TrafficLightDetector(debug=True)
        self.lane_controller = LaneController(debug=True)
        self.state = 'normal'
        self.green_light_detected = False  # Flag to track green light detection
        self.debug_seq = 0  # Sequence number for debugging

    def run(self):
        """
        Main loop to run the robot's operations. It captures images, detects traffic lights, and controls the robot's movements.
        """
        while self.tick():
            image = self.get_camera_image()

            if image is None:
                continue

            self.debug_seq += 1  # Increment debug sequence number

            # Convert image to BGR format if necessary.
            if image.shape[2] == 4:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detect the traffic light color.
            traffic_light_color = self.traffic_light_detector.detect(image_bgr)
            if self.traffic_light_detector.debug:
                print(f'{self.debug_seq}: Traffic light color: {traffic_light_color}')

            # Update the robot state based on the traffic light color.
            if traffic_light_color == 'red' or traffic_light_color == 'yellow':
                if not self.green_light_detected:
                    self.state = 'stop'
            elif traffic_light_color == 'green':
                self.green_light_detected = True
                self.state = 'moving'
            elif self.green_light_detected:
                self.state = 'moving'

            # Print the robot state for debugging.
            if self.traffic_light_detector.debug:
                print(f'{self.debug_seq}: Robot state: {self.state}')

            # Control the robot's movement based on the state.
            if self.state == 'stop':
                steering_angle, speed = 0, 0
                if self.traffic_light_detector.debug:
                    print(f'{self.debug_seq}: Steering angle: {steering_angle}, Speed: {speed} (stop)')
            else:
                steering_angle, speed = self.lane_controller.get_angle_and_speed(image)
                if self.traffic_light_detector.debug:
                    print(f'{self.debug_seq}: Steering angle: {steering_angle}, Speed: {speed} (moving)')

            self.set_steering_angle(steering_angle)
            self.set_speed(speed)

            # If at a crossroads, randomly decide the next action.
            if self.lane_controller.mode == 'driving' and self.lane_controller._at_crossroads(self.lane_controller._get_road(image)):
                self.lane_controller.mode = random.choice(['turning_left', 'turning_right', 'driving'])
                if self.traffic_light_detector.debug:
                    print(f'{self.debug_seq}: Crossroads detected! Changing mode to {self.lane_controller.mode}')

            # Debugging output for the robot's movement direction.
            if self.traffic_light_detector.debug:
                if self.lane_controller.mode == 'turning_left':
                    print(f'{self.debug_seq}: Robot Turning left')
                elif self.lane_controller.mode == 'turning_right':
                    print(f'{self.debug_seq}: Robot Turning right')
                else:
                    print(f'{self.debug_seq}: Robot Moving forward')

if __name__ == '__main__':
    robot = MyRobot()
    robot.run()

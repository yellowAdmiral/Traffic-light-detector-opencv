import numpy as np
import cv2
import random
from rasrobot import RASRobot

class TrafficLightDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.debug_seq = 0

    def detect(self, image):
        if image is None:
            return 'none'

        self.debug_seq += 1

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        green_lower = np.array([35, 100, 100])
        green_upper = np.array([90, 255, 255])

        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        blurred_red = cv2.GaussianBlur(mask_red, (9, 9), 2)
        blurred_yellow = cv2.GaussianBlur(mask_yellow, (9, 9), 2)
        blurred_green = cv2.GaussianBlur(mask_green, (9, 9), 2)

        def find_contours(mask, color):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 20:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.8 <= aspect_ratio <= 1.2:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if self.debug:
                            print(f"{self.debug_seq}: {color.capitalize()} light detected with area {area} at ({x}, {y}, {w}, {h}).")
                        return True
            return False

        height, width = image.shape[:2]
        roi_height_start = 0
        roi_height_end = height
        roi_width_start = width // 2
        roi_width_end = width

        roi = image[roi_height_start:roi_height_end, roi_width_start:roi_width_end]

        if find_contours(blurred_red[roi_height_start:roi_height_end, roi_width_start:roi_width_end], 'red'):
            return "red"
        elif find_contours(blurred_yellow[roi_height_start:roi_height_end, roi_width_start:roi_width_end], 'yellow'):
            return "yellow"
        elif find_contours(blurred_green[roi_height_start:roi_height_end, roi_width_start:roi_width_end], 'green'):
            return "green"
        else:
            return "none"

class LaneController:
    def __init__(self, debug=False):
        self.mode = 'driving'
        self.turn_counter = 0

        self.steering_row = 94
        self.crossroad_row = 90

        self.max_steering_angle = 0.3
        self.max_speed = 30  # Reduced speed for better control
        self.min_speed = 20  # Reduced speed for better control

        self.debug = debug
        self.traffic_light_state = 'unknown'

    def _display_image(self, image, name, scale=2):
        if not self.debug:
            return

        image = image.copy()
        image[self.steering_row, :] = 1.0 - image[self.steering_row, :]
        image[self.crossroad_row, :] = 1.0 - image[self.crossroad_row, :]

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, image.shape[1] * scale, image.shape[0] * scale)
        cv2.imshow(name, image)
        cv2.waitKey(1)

    def _get_road(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_tarmac = np.array([0, 0, 0], np.uint8)
        upper_tarmac = np.array([200, 50, 100], np.uint8)
        mask_tarmac = cv2.inRange(hsv, lower_tarmac, upper_tarmac).astype(bool)

        lower_yellow = np.array([25, 100, 100], np.uint8)
        upper_yellow = np.array([50, 255, 255], np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow).astype(bool)

        binary_image = np.zeros(image.shape)
        binary_image[mask_tarmac] = 1.0
        binary_image[mask_yellow] = 1.0
        binary_image = binary_image[:, :, 0]

        reduce_noise = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, (3, 3))

        self._display_image(reduce_noise, 'road image')

        return reduce_noise

    def _get_road_center(self, road_image):
        road_img_line = road_image[self.steering_row, :]
        if road_img_line.sum() == 0:
            return None

        idx = road_img_line.nonzero()[0]
        cx = np.mean(idx / len(road_img_line))
        center_normalised = (cx - 0.5) * 2

        return center_normalised

    def _at_crossroads(self, road_image):
        steering_sum = road_image[self.steering_row, :].sum()
        crossroad_sum = road_image[self.crossroad_row, :].sum()
        diff = crossroad_sum - steering_sum

        if diff > 50 and crossroad_sum > 200:
            return True
        return False

    def _get_steering_angle(self, road_center):
        if road_center is None:
            return 0

        preferred_road_center_pos = -0.15
        angle = road_center - preferred_road_center_pos
        angle = np.clip(angle, -self.max_steering_angle, self.max_steering_angle)

        return angle

    def _get_speed(self, steering_angle):
        speed_factor = (self.max_steering_angle - abs(steering_angle)) / self.max_steering_angle
        speed = (self.max_speed - self.min_speed) * speed_factor + self.min_speed

        return speed

    def get_angle_and_speed(self, image):
        traffic_light_color = self._detect_traffic_light(image)

        if traffic_light_color == 'red':
            self.traffic_light_state = 'stop'
        elif traffic_light_color == 'yellow':
            self.traffic_light_state = 'stop'
        elif traffic_light_color == 'green':
            self.traffic_light_state = 'normal'

        if self.traffic_light_state == 'stop':
            if self.debug:
                print(f'Traffic light color: {traffic_light_color}, Robot state: stop')
            return 0, 0

        steering_angle = 0

        if self.mode == 'turning_left':
            self.turn_counter += 1
            if self.turn_counter < 50:
                steering_angle = 0
            elif self.turn_counter < 130:
                steering_angle = -self.max_steering_angle
            else:
                self.mode = 'driving'
                self.turn_counter = 0

        elif self.mode == 'turning_right':
            self.turn_counter += 1
            if self.turn_counter < 50:
                steering_angle = 0
            elif self.turn_counter < 130:
                steering_angle = self.max_steering_angle
            else:
                self.mode = 'driving'
                self.turn_counter = 0

        elif self.mode == 'driving':
            road_image = self._get_road(image)
            road_center = self._get_road_center(road_image)

            if road_center is None:
                steering_angle = 0
            else:
                steering_angle = self._get_steering_angle(road_center)

            if self._at_crossroads(road_image):
                self.mode = np.random.choice(['turning_left', 'turning_right'])
                if self.debug:
                    print(f'Crossroads detected! Changing mode to {self.mode}')

        speed = self._get_speed(steering_angle)

        if self.debug:
            print(f'Traffic light color: {traffic_light_color}, Robot state: normal, Steering angle: {steering_angle}, Speed: {speed}')

        return steering_angle, speed

class MyRobot(RASRobot):
    def __init__(self):
        super(MyRobot, self).__init__()
        self.traffic_light_detector = TrafficLightDetector(debug=True)
        self.lane_controller = LaneController(debug=True)
        self.state = 'normal'
        self.green_light_detected = False
        self.debug_seq = 0

    def run(self):
        while self.tick():
            image = self.get_camera_image()

            if image is None:
                continue

            self.debug_seq += 1

            if image.shape[2] == 4:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            traffic_light_color = self.traffic_light_detector.detect(image_bgr)
            if self.traffic_light_detector.debug:
                print(f'{self.debug_seq}: Traffic light color: {traffic_light_color}')

            if traffic_light_color == 'red' or traffic_light_color == 'yellow':
                if not self.green_light_detected:
                    self.state = 'stop'
            elif traffic_light_color == 'green':
                self.green_light_detected = True
                self.state = 'moving'
            elif self.green_light_detected:
                self.state = 'moving'

            if self.traffic_light_detector.debug:
                print(f'{self.debug_seq}: Robot state: {self.state}')

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

            if self.lane_controller.mode == 'driving' and self.lane_controller._at_crossroads(self.lane_controller._get_road(image)):
                self.lane_controller.mode = random.choice(['turning_left', 'turning_right', 'driving'])
                if self.traffic_light_detector.debug:
                    print(f'{self.debug_seq}: Crossroads detected! Changing mode to {self.lane_controller.mode}')

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

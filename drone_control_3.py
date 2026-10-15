import airsim
import cv2
import numpy as np
import tkinter as tk
from tkinter import Button, Label, StringVar
import time
import threading
from collections import deque  # For marker position buffer

class DroneController:
    def __init__(self, master):
        self.master = master
        self.master.title("Drone Controller with ArUco Marker Detection")

        # Initialize AirSim client
        self.client = airsim.MultirotorClient()
        self.status_var = StringVar()
        self.position_var = StringVar()
        self.setup_connection()

        # UI Setup
        self.status_label = Label(master, textvariable=self.status_var)
        self.status_label.pack()
        # self.position_label = Label(master, textvariable=self.position_var)
        # self.position_label.pack()

        self.takeoff_button = Button(master, text="Take Off", command=self.take_off)
        self.takeoff_button.pack()
        self.land_button = Button(master, text="Land", command=self.land)
        self.land_button.pack()

        # self.display_current_position()

        # Keyboard Controls
        master.bind('<w>', self.move_forward)
        master.bind('<s>', self.move_backward)
        master.bind('<a>', self.move_left)
        master.bind('<d>', self.move_right)
        master.bind('<q>', self.move_up)
        master.bind('<e>', self.move_down)

        # ArUco Detection Thread
        self.run_detection = True
        self.marker_detected = False  # Flag to stop detection after stabilization
        self.marker_position = None  # To store the marker's final position
        self.position_buffer = deque(maxlen=5)  # Buffer to store last 5 positions

        self.aruco_thread = threading.Thread(target=self.aruco_detection_loop)
        self.aruco_thread.start()

    def setup_connection(self):
        try:
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.status_var.set("Status: Connected and ready")
        except Exception as e:
            self.status_var.set(f"Status: Failed to connect to AirSim: {str(e)}")

    def display_current_position(self):
        pos = self.client.simGetGroundTruthKinematics().position
        self.position_var.set(f"Position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={-pos.z_val:.2f}")
        self.master.after(1000, self.display_current_position)

    def take_off(self):
        try:
            self.client.takeoffAsync().join()
            self.status_var.set("Status: Takeoff successful")
        except Exception as e:
            self.status_var.set(f"Status: Takeoff failed: {str(e)}")

    def land(self):
        try:
            self.run_detection = False  # Stop ArUco detection loop
            time.sleep(0.5)  # Give the thread time to exit

            self.client.landAsync().join()
            self.status_var.set("Status: Landing successful")
        except Exception as e:
            self.status_var.set(f"Status: Landing failed: {str(e)}")

    def move_forward(self, event):
        self.client.moveByVelocityAsync(3, 0, 0, 1)
        self.status_var.set("Moving forward")

    def move_backward(self, event):
        self.client.moveByVelocityAsync(-3, 0, 0, 1)
        self.status_var.set("Moving backward")

    def move_left(self, event):
        self.client.moveByVelocityAsync(0, -3, 0, 1)
        self.status_var.set("Moving left")

    def move_right(self, event):
        self.client.moveByVelocityAsync(0, 3, 0, 1)
        self.status_var.set("Moving right")

    def move_up(self, event):
        self.client.moveByVelocityAsync(0, 0, -3, 1)
        self.status_var.set("Moving up")

    def move_down(self, event):
        self.client.moveByVelocityAsync(0, 0, 3, 1)
        self.status_var.set("Moving down")

    def detect_aruco_marker(self, image):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        debug_img = image.copy()

        camera_matrix = np.array([[128, 0, 128], [0, 72, 72], [0, 0, 1]])
        dist_coeffs = np.zeros((5, 1))
        marker_size = 1.0  # Marker size in meters

        if ids is not None:
            for corner in corners:
                retval, rvec, tvec = cv2.solvePnP(
                    np.array([[0, 0, 0], [marker_size, 0, 0], 
                              [marker_size, marker_size, 0], [0, marker_size, 0]], dtype=np.float32),
                    corner, camera_matrix, dist_coeffs
                )

                cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
                self.position_buffer.append(tvec.flatten())
                # print(self.position_buffer)

                # Check if the position is stable
                if self.is_position_stable():
                    self.marker_position = tvec.flatten()
                    self.marker_detected = True  # Stop further detection
                    self.status_var.set("Marker stabilized. Moving to marker position...")
                    self.adjust_position_based_on_marker()

        else:
            print("No ArUco marker detected.")

        cv2.imshow("Markers", debug_img)
        cv2.waitKey(1)

    def is_position_stable(self):
        if len(self.position_buffer) < 5:
            return False

        buffer_array = np.array(self.position_buffer)
        std_dev = np.std(buffer_array, axis=0)

        stability_threshold = 0.05  # 5 cm deviation allowed
        return np.all(std_dev < stability_threshold)

    def adjust_position_based_on_marker(self):
        if self.marker_position is None:
            return

        x_error, y_error, z_error = self.marker_position
        # threshold = 0.30  # 30 cm threshold
        print(x_error, y_error, z_error)

    
        drone_position = self.client.simGetGroundTruthKinematics().position
        print(f"Current Drone Position: X={drone_position.x_val:.2f}, "
            f"Y={drone_position.y_val:.2f}, Z={-drone_position.z_val:.2f}")

        
        self.client.moveByVelocityAsync(-x_error, 0, 0, 1).join()
        self.client.moveByVelocityAsync(0, -y_error, 0, 1).join()

        self.status_var.set("Aligned with marker. Landing...")
        self.land()
        drone_position = self.client.simGetGroundTruthKinematics().position
        print(f"Current Drone Position: X={drone_position.x_val:.2f}, "
            f"Y={drone_position.y_val:.2f}, Z={-drone_position.z_val:.2f}")

    def process_camera(self, camera_id):
        raw_image = self.client.simGetImage(camera_id, airsim.ImageType.Scene)
        if raw_image:
            img = cv2.imdecode(airsim.string_to_uint8_array(raw_image), cv2.IMREAD_COLOR)
            if img is not None:
                self.detect_aruco_marker(img)

    def aruco_detection_loop(self):
        while self.run_detection and not self.marker_detected:
            try:
                self.process_camera("3")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in ArUco detection loop: {str(e)}")
                break

    def on_close(self):
        self.run_detection = False
        if self.aruco_thread.is_alive():
            self.aruco_thread.join()
        cv2.destroyAllWindows()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = DroneController(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == '__main__':
    main()

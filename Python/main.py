"""
Real-time Leap Motion Infrared Processing Pipeline with ASL Recognition
Based on your working LeapMotionProcessor with ONNX model integration
"""

import numpy as np
import cv2
import struct
import pickle
import matplotlib.pyplot as plt
from collections import deque
from unity_sender import UnitySender
import time
import onnxruntime as ort


class LeapMotionASLProcessor:

    def __init__(self, distortion_map_path='./distortion_map.p', model_path='infrared_asl_cnn.onnx'):
        # Camera setup
        self.cam = None
        self.w = 1024
        self.h = 1024

        # Frame tracking
        self.frame_count_bright = 0
        self.frame_count_dark = 0

        # Storage for recent frames
        self.recent_bright_frames = deque(maxlen=10)
        self.recent_dark_frames = deque(maxlen=10)

        # Load distortion correction maps
        self.load_distortion_maps(distortion_map_path)

        # Load ONNX model
        self.load_onnx_model(model_path)

        # ASL class mapping (from your training code)
        self.sign_mapping = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q',
            16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
        }

        # Processing flags
        self.save_frames = False
        self.show_processing_steps = False
        self.enable_recognition = True

        # Recognition parameters
        self.prediction_interval = 3  # Predict every N frames for performance
        self.frame_counter = 0
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_second_prediction = None
        self.current_second_confidence = 0.0

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)
        self.confidence_threshold = 0.7

    def load_distortion_maps(self, data_file):
        try:
            with open(data_file, mode='rb') as f:
                data = pickle.load(f, encoding="latin1")
                self.left_coordinates = data['left_coordinates']
                self.left_coefficients = data['left_coefficients']
                self.right_coordinates = data['right_coordinates']
                self.right_coefficients = data['right_coefficients']
        except FileNotFoundError:
            self.left_coordinates = None
            self.left_coefficients = None
            self.right_coordinates = None
            self.right_coefficients = None

    def load_onnx_model(self, model_path):
        try:
            self.onnx_session = ort.InferenceSession(model_path)
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_name = self.onnx_session.get_outputs()[0].name

            # Get model input shape
            input_shape = self.onnx_session.get_inputs()[0].shape
            print(f"   Input name: {self.input_name}")
            print(f"   Input shape: {input_shape}")
            print(f"   Output name: {self.output_name}")

        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.onnx_session = None

    def initialize_camera(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_GAIN, 10)
        #self.cam.set(cv2.CAP_PROP_EXPOSURE, -10)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Important for infrared!

        actual_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Requested: {self.w}x{self.h}, Actual: {actual_width}x{actual_height}")

        if not self.cam.isOpened():
            raise Exception("Could not open Leap Motion camera")

        print("✅ Camera initialized successfully")
        return True

    def undistort(self, image, coordinate_map, coefficient_map, width=640, height=640):
        if coordinate_map is None or coefficient_map is None:
            # Skip distortion correction if maps not available
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Apply distortion correction
        destination = cv2.remap(image,
                                coordinate_map,
                                coefficient_map,
                                interpolation=cv2.INTER_LINEAR)

        # Resize to desired output size
        destination = cv2.resize(destination,
                                 (width, height),
                                 interpolation=cv2.INTER_LINEAR)
        return destination

    def hand_cropping(self, img):
        # Initial crop to center region
        img = img[0:400, 0:400]

        # Find hand boundaries using intensity projection
        dist_x = np.sum(img, 0)  # Horizontal projection
        dist_y = np.sum(img, 1)  # Vertical projection

        # Find horizontal hand span
        span_x = np.where(dist_x > 500)
        if len(span_x[0]) == 0:
            return None  # No hand detected

        span_x_start = np.min(span_x)
        span_x_end = np.max(span_x)

        # Find vertical hand span
        span_y = np.where(dist_y > 50)
        if len(span_y[0]) == 0:
            return None  # No hand detected

        span_y_start = np.min(span_y)
        span_y_end = np.max(span_y)

        # Aspect ratio correction
        if len(span_y[0]) / len(span_x[0]) > 2:
            span_y_end = int(span_y_start + len(span_x[0]) * 1.8)

        # Extract final hand region
        return img[span_y_start:span_y_end + 1, span_x_start:span_x_end + 1]

    def resize_img(self, img, target_size):
        if len(img.shape) == 3:
            img = img.squeeze()  # Remove channel dimension if present

        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    def process_frame_complete(self, raw_frame, use_left_camera=True):
        try:
            # Step 1: Apply distortion correction
            if use_left_camera:
                processed_img = self.undistort(raw_frame,
                                               self.left_coordinates,
                                               self.left_coefficients,
                                               640, 640)
            else:
                processed_img = self.undistort(raw_frame,
                                               self.right_coordinates,
                                               self.right_coefficients,
                                               640, 640)

            # Step 2: Flip image (same as training)
            processed_img = cv2.flip(processed_img, 1)

            # Step 3: Apply threshold to remove background
            img_thresh = np.copy(processed_img)
            img_thresh[img_thresh < 60] = 0

            # Step 4: Apply smart cropping
            cropped_hand = self.hand_cropping(img_thresh)

            if cropped_hand is None:
                return None, None

            # Step 5: Resize to final size (32x32 for model)
            final_img = self.resize_img(cropped_hand, 32)

            # Step 6: Normalize and add channel dimension
            final_img = final_img.astype(np.float32) / 255.0
            final_img = np.expand_dims(final_img, axis=-1)

            # Return both the final processed image and intermediate steps for visualization
            processing_steps = {
                'original': raw_frame,
                'undistorted': processed_img,
                'thresholded': img_thresh,
                'cropped': cropped_hand,
                'final': final_img
            }

            return final_img, processing_steps

        except Exception as e:
            print(f"Processing error: {e}")
            return None, None

    def predict_asl_sign(self, processed_image):
        if self.onnx_session is None or processed_image is None:
            return None, 0.0, None, 0.0

        try:
            # Prepare input for ONNX model
            # Add batch dimension: (1, 32, 32, 1)
            input_data = np.expand_dims(processed_image, axis=0)
            #input_data = input_data.transpose((0, 3, 1, 2))

            # Run inference
            outputs = self.onnx_session.run([self.output_name], {self.input_name: input_data})
            predictions = outputs[0][0]  # Get the prediction array

            # Get top 2 predictions
            top_2_indices = np.argsort(predictions)[-2:][::-1]  # Top 2 in descending order

            # First prediction
            first_idx = top_2_indices[0]
            first_confidence = predictions[first_idx]
            first_letter = self.sign_mapping[first_idx]

            # Second prediction
            second_idx = top_2_indices[1]
            second_confidence = predictions[second_idx]
            second_letter = self.sign_mapping[second_idx]

            return first_letter, first_confidence, second_letter, second_confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None, 0.0

    def smooth_predictions(self, prediction, confidence, second_prediction, second_confidence):
        if prediction is None:
            return None, 0.0, None, 0.0

        # Add to history (only track primary prediction for smoothing)
        self.prediction_history.append((prediction, confidence))

        # If we have enough samples, find the most confident prediction
        if len(self.prediction_history) >= 3:
            # Get the most frequent prediction with high confidence
            high_conf_predictions = [(p, c) for p, c in self.prediction_history if c > self.confidence_threshold]

            if high_conf_predictions:
                # Find most frequent high-confidence prediction
                from collections import Counter
                pred_counts = Counter([p for p, c in high_conf_predictions])
                most_common = pred_counts.most_common(1)[0]

                if most_common[1] >= 2:  # At least 2 occurrences
                    # Get average confidence for this prediction
                    avg_conf = np.mean([c for p, c in high_conf_predictions if p == most_common[0]])
                    return most_common[0], avg_conf, second_prediction, second_confidence

        # Return current predictions if smoothing doesn't apply
        return prediction, confidence, second_prediction, second_confidence

    def run_realtime_processing(self):
        if not self.initialize_camera():
            return

        self.unity = UnitySender()

        # Create display windows
        #cv2.namedWindow('Bright Frames', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Dark Frames', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Hand', cv2.WINDOW_NORMAL)

        use_left_camera = True
        once = True
        save_counter = 0

        try:
            while True:
                # Read frame
                ret, frame = self.cam.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Reshape frame (exactly as your working version)
                frame = np.reshape(frame, (self.h, int(frame.size / self.h)))

                # Extract embedded parameters
                embedded_line = frame[-1, :68]
                embedded_params = struct.unpack("IIIIHHHHIQIIIIIII", embedded_line.tobytes())
                frame_label = embedded_params[1]

                # Crop frame (exactly as your working version: frame[:, 1024:2048])
                frame = frame[:, 1024:2048]
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if once:
                    once = False
                    print(f"Frame shape: {frame.shape}")

                # Resize for display (exactly as your working version)
                display_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)

                # Rotate frame for processing (exactly as your working version)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Process for hand detection
                processed_img, steps = self.process_frame_complete(frame, use_left_camera)

                if (self.enable_recognition and
                    processed_img is not None and
                    self.frame_counter % self.prediction_interval == 0):

                    pred, conf, second_pred, second_conf = self.predict_asl_sign(processed_img)
                    if pred is not None:
                        # Apply smoothing
                        current_pred, current_conf, current_second_pred, current_second_conf = self.smooth_predictions(
                            pred, conf, second_pred, second_conf)
                        self.current_prediction = current_pred
                        self.current_confidence = current_conf
                        self.current_second_prediction = current_second_pred
                        self.current_second_confidence = current_second_conf

                        if self.unity is not None:
                            try:
                                self.unity.send(current_pred, current_conf, current_second_pred, current_second_conf)
                            except Exception as e:
                                print(f"Unity send failed: {e}")

                # Process bright/dark frames (exactly as your working version)
                # if frame_label == 0:
                #     # Bright frame
                #     self.frame_count_bright += 1

                    # Add ASL prediction to display
                    if self.enable_recognition and self.current_prediction:
                        cv2.putText(display_frame,
                                    f"1st: {self.current_prediction} ({self.current_confidence:.2f})",
                                    (5, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if self.current_second_prediction:
                            cv2.putText(display_frame,
                                        f"2nd: {self.current_second_prediction} ({self.current_second_confidence:.2f})",
                                        (5, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                else:
                    # Dark frame
                    self.frame_count_dark += 1

                    # Add ASL prediction to display
                    if self.enable_recognition and self.current_prediction:
                        cv2.putText(display_frame,
                                    f"1st: {self.current_prediction} ({self.current_confidence:.2f})",
                                    (5, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if self.current_second_prediction:
                            cv2.putText(display_frame,
                                        f"2nd: {self.current_second_prediction} ({self.current_second_confidence:.2f})",
                                        (5, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # cv2.putText(display_frame,
                    #             f"dark frame count: {self.frame_count_dark}",
                    #             (5, display_frame.shape[0] - 5),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    cv2.imshow("Dark Frames", display_frame)

                    # Store recent frame
                    self.recent_dark_frames.append(frame.copy())

                # Display processed hand if detected (exactly as your working version)
                if processed_img is not None:
                    # Scale up for better visibility
                    hand_display = (processed_img.squeeze() * 255).astype(np.uint8)
                    hand_display = cv2.resize(hand_display, (200, 400), interpolation=cv2.INTER_NEAREST)

                    # Add prediction text to hand display
                    if self.enable_recognition and self.current_prediction:
                        hand_display_color = cv2.cvtColor(hand_display, cv2.COLOR_GRAY2RGB)
                        cv2.putText(hand_display_color,
                                    f"1st: {self.current_prediction}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(hand_display_color,
                                    f"{self.current_confidence:.2f}",
                                    (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        if self.current_second_prediction:
                            cv2.putText(hand_display_color,
                                        f"2nd: {self.current_second_prediction}",
                                        (10, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            cv2.putText(hand_display_color,
                                        f"{self.current_second_confidence:.2f}",
                                        (10, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.imshow('Processed Hand', hand_display_color)
                    else:
                        cv2.imshow('Processed Hand', hand_display)

                    # Show processing steps if enabled
                    if self.show_processing_steps and steps is not None:
                        self.display_processing_steps(steps)

                    # Save if enabled
                    if self.save_frames:
                        save_path = f'processed_hand_{save_counter:04d}.npy'
                        np.save(save_path, processed_img)
                        save_counter += 1
                        if save_counter % 10 == 0:
                            print(f"Saved {save_counter} processed frames")

                self.frame_counter += 1

                # Handle keyboard input (same as your working version + ASL controls)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    processor = LeapMotionASLProcessor()
    processor.run_realtime_processing()


if __name__ == "__main__":
    main()
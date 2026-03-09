"""
Real-time Leap Motion Infrared Processing Pipeline
Combines your camera access code with the preprocessing pipeline
"""

import numpy as np
import cv2
import struct
import pickle
import matplotlib.pyplot as plt
from collections import deque
import time


class LeapMotionProcessor:
    """
    Real-time Leap Motion infrared image processor
    """

    def __init__(self, distortion_map_path='./distortion_map.p'):
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

        # Processing flags
        self.save_frames = False
        self.show_processing_steps = False

    def load_distortion_maps(self, data_file):
        """Load distortion correction maps"""
        try:
            with open(data_file, mode='rb') as f:
                data = pickle.load(f)
                self.left_coordinates = data['left_coordinates']
                self.left_coefficients = data['left_coefficients']
                self.right_coordinates = data['right_coordinates']
                self.right_coefficients = data['right_coefficients']
            print("✅ Distortion maps loaded successfully")
        except FileNotFoundError:
            print("⚠️  Distortion map file not found. Processing will continue without distortion correction.")
            self.left_coordinates = None
            self.left_coefficients = None
            self.right_coordinates = None
            self.right_coefficients = None

    def initialize_camera(self):
        """Initialize Leap Motion camera"""
        print("🎥 Initializing Leap Motion camera...")

        self.cam = cv2.VideoCapture(1)
        #self.cam.set(cv2.CAP_PROP_GAIN, 8)
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
        """Apply distortion correction (same as your training pipeline)"""
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
        """Smart hand cropping (same as your training pipeline)"""
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
        """Resize image to target size (32x32 for your model)"""
        if len(img.shape) == 3:
            img = img.squeeze()  # Remove channel dimension if present

        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    def process_frame_complete(self, raw_frame, use_left_camera=True):
        """
        Complete processing pipeline from raw frame to model-ready image
        """
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

    def separate_stereo_frames(self, combined_frame):
        """
        Separate the combined stereo frame into left and right images
        Based on your camera setup
        """
        h, w = combined_frame.shape

        # Assuming the frame contains both left and right images
        # You might need to adjust this based on your specific camera setup

        if w > h:  # Side-by-side format
            mid = w // 2
            left_frame = combined_frame[:, :mid]
            right_frame = combined_frame[:, mid:]
        else:  # Top-bottom format or single camera
            # If it's a single combined frame, you might need different logic
            # For now, assume it's the format your training data expects
            left_frame = combined_frame
            right_frame = combined_frame

        return left_frame, right_frame

    def run_realtime_processing(self):
        """
        Main loop for real-time processing
        """
        if not self.initialize_camera():
            return

        print("\n🚀 Starting real-time processing...")
        print("Controls:")
        print("  - 'q': Quit")
        print("  - 's': Toggle saving frames")
        print("  - 'p': Toggle processing steps display")
        print("  - 'l': Use left camera")
        print("  - 'r': Use right camera")
        print("  - 'i': Show frame info")

        # Create display windows
        cv2.namedWindow('Bright Frames', cv2.WINDOW_NORMAL)
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

                # Reshape frame
                frame = np.reshape(frame, (self.h, int(frame.size / self.h)))

                # Extract embedded parameters
                embedded_line = frame[-1, :68]
                embedded_params = struct.unpack("IIIIHHHHIQIIIIIII", embedded_line.tobytes())
                frame_label = embedded_params[1]
                frame = frame[:, 1024:2048]

                if once:
                    once = False
                    print(f"Frame shape: {frame.shape}")

                # Resize for display
                display_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # Process bright/dark frames
                if frame_label == 0:
                    # Bright frame
                    self.frame_count_bright += 1
                    cv2.putText(display_frame,
                                f"bright frame count: {self.frame_count_bright}",
                                (5, display_frame.shape[0] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    cv2.imshow("Bright Frames", display_frame)

                    # Store recent frame
                    self.recent_bright_frames.append(frame.copy())

                    # Process for hand detection
                    processed_img, steps = self.process_frame_complete(frame, use_left_camera)

                else:
                    # Dark frame
                    self.frame_count_dark += 1
                    cv2.putText(display_frame,
                                f"dark frame count: {self.frame_count_dark}",
                                (5, display_frame.shape[0] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                    cv2.imshow("Dark Frames", display_frame)

                    # Store recent frame
                    self.recent_dark_frames.append(frame.copy())

                    # Process for hand detection
                    processed_img, steps = self.process_frame_complete(frame, use_left_camera)

                # Display processed hand if detected
                if processed_img is not None:
                    # Scale up for better visibility
                    hand_display = (processed_img.squeeze() * 255).astype(np.uint8)
                    hand_display = cv2.resize(hand_display, (200, 400), interpolation=cv2.INTER_NEAREST)
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

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frames = not self.save_frames
                    print(f"Frame saving: {'ON' if self.save_frames else 'OFF'}")
                elif key == ord('p'):
                    self.show_processing_steps = not self.show_processing_steps
                    print(f"Processing steps display: {'ON' if self.show_processing_steps else 'OFF'}")
                elif key == ord('l'):
                    use_left_camera = True
                    print("Using LEFT camera")
                elif key == ord('r'):
                    use_left_camera = False
                    print("Using RIGHT camera")
                elif key == ord('i'):
                    print(f"\n📊 Frame Info:")
                    print(f"  Bright frames: {self.frame_count_bright}")
                    print(f"  Dark frames: {self.frame_count_dark}")
                    print(f"  Using: {'LEFT' if use_left_camera else 'RIGHT'} camera")
                    print(f"  Saving: {'ON' if self.save_frames else 'OFF'}")
                    print(f"  Saved frames: {save_counter}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def display_processing_steps(self, steps):
        """Display processing steps for debugging"""
        if steps is None:
            return

        # Create a combined view of processing steps
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        if 'undistorted' in steps:
            axes[0].imshow(steps['undistorted'], cmap='gray')
            axes[0].set_title('Undistorted')
            axes[0].axis('off')

        if 'thresholded' in steps:
            axes[1].imshow(steps['thresholded'], cmap='gray')
            axes[1].set_title('Thresholded')
            axes[1].axis('off')

        if 'cropped' in steps:
            axes[2].imshow(steps['cropped'], cmap='gray')
            axes[2].set_title('Cropped')
            axes[2].axis('off')

        if 'final' in steps:
            axes[3].imshow(steps['final'].squeeze(), cmap='gray')
            axes[3].set_title('Final (32x32)')
            axes[3].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def cleanup(self):
        """Cleanup resources"""
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")


def main():
    """Main function"""
    processor = LeapMotionProcessor()
    processor.run_realtime_processing()


if __name__ == "__main__":
    main()
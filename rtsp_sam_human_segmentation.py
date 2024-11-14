import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import logging
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTSPHumanSegmenter:
    def __init__(self, rtsp_url, sam_checkpoint='sam_vit_h_4b8939.pth', device='cuda', process_every_n_frames=30, process_width=640, process_height=480, confidence_threshold=0.7):
        """Initialize the RTSP Human Segmentation pipeline"""
        # Set up device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Store processing frame size
        self.process_width = process_width
        self.process_height = process_height
        self.original_width = None
        self.original_height = None
        
        # Detection parameters
        self.confidence_threshold = confidence_threshold
        logger.info(f"YOLO confidence threshold set to: {confidence_threshold}")
        
        # Frame processing control
        self.process_every_n_frames = process_every_n_frames
        self.frame_count = 0
        self.last_processed_frame = None
        
        # Store RTSP URL
        self.rtsp_url = rtsp_url
        self.cap = None
        
        try:
            # Initialize YOLO for fast human detection
            self.yolo_model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully")
            
            # Initialize SAM
            self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            self.sam_model.to(device=self.device)
            self.predictor = SamPredictor(self.sam_model)
            logger.info("SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
        
        # Connect to stream and get original dimensions
        self.connect_to_stream()
        
    def scale_coordinates(self, box, from_size, to_size):
        """Scale coordinates from one resolution to another"""
        x1, y1, x2, y2 = box
        from_width, from_height = from_size
        to_width, to_height = to_size
        
        # Calculate scale factors
        width_scale = to_width / from_width
        height_scale = to_height / from_height
        
        # Scale coordinates
        x1_scaled = int(x1 * width_scale)
        y1_scaled = int(y1 * height_scale)
        x2_scaled = int(x2 * width_scale)
        y2_scaled = int(y2 * height_scale)
        
        return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
    
    def scale_mask(self, mask, from_size, to_size):
        """Scale a binary mask from one resolution to another"""
        return cv2.resize(mask.astype(np.uint8), (to_size[0], to_size[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    def apply_visualizations(self, original_frame, processed_frame, box, mask, color):
        """Apply both bounding box and segmentation mask to the high-res image"""
        try:
            # Create a copy of the original frame
            output = original_frame.copy()
            
            # Scale the mask to original resolution
            scaled_mask = self.scale_mask(
                mask,
                (self.process_width, self.process_height),
                (self.original_width, self.original_height)
            )
            
            # Scale the box coordinates
            scaled_box = self.scale_coordinates(
                box,
                (self.process_width, self.process_height),
                (self.original_width, self.original_height)
            )
            
            # Apply segmentation mask
            colored_mask = np.zeros_like(output)
            colored_mask[scaled_mask] = color
            output = cv2.addWeighted(output, 1, colored_mask, 0.5, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, scaled_box)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            conf_text = f"Person"
            cv2.putText(output, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return output
            
        except Exception as e:
            logger.error(f"Error applying visualizations: {str(e)}")
            return original_frame
    
    def connect_to_stream(self, max_retries=3):
        """Attempt to connect to the RTSP stream with retries"""
        for attempt in range(max_retries):
            try:
                if self.cap is not None:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if self.cap.isOpened():
                    logger.info("RTSP stream opened successfully")
                    
                    # Get original dimensions
                    self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"Original frame size: {self.original_width}x{self.original_height}")
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
        
        logger.error(f"Failed to open RTSP stream after {max_retries} attempts")
        raise ValueError(f"Could not open RTSP stream: {self.rtsp_url}")
    
    def process_frame(self):
        """Process a single frame from the RTSP stream"""
        try:
            ret, original_frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from stream")
                # Try to reconnect to stream
                if self.connect_to_stream():
                    return self.last_processed_frame if self.last_processed_frame is not None else None
                return None
            
            # Create a processed (downscaled) copy for detection
            processed_frame = cv2.resize(original_frame, (self.process_width, self.process_height))
            
            self.frame_count += 1
            
            # Only process every nth frame
            if self.frame_count % self.process_every_n_frames == 0:
                # Detect humans using YOLO with confidence threshold
                results = self.yolo_model(processed_frame, classes=[0], conf=self.confidence_threshold)  # class 0 is person in COCO
                
                # If no humans detected, return original frame
                if len(results[0].boxes) == 0:
                    self.last_processed_frame = original_frame
                    return original_frame
                
                logger.info(f"Detected {len(results[0].boxes)} humans in frame with confidence >= {self.confidence_threshold}")
                
                # Set image for SAM
                self.predictor.set_image(processed_frame)
                
                # Process each detected person
                output_frame = original_frame.copy()
                for i, box in enumerate(results[0].boxes.xyxy):
                    try:
                        # Convert box coordinates to the format SAM expects
                        box_coords = box.cpu().numpy().astype(float)
                        
                        # Create input box in format [[x0, y0, x1, y1]]
                        input_box = np.array([[
                            box_coords[0], box_coords[1],
                            box_coords[2], box_coords[3]
                        ]])
                        
                        # Generate mask for the person
                        masks, _, _ = self.predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box,
                            multimask_output=False
                        )
                        
                        # Apply visualizations if mask was generated
                        if masks.shape[0] > 0:
                            color = np.random.randint(0, 255, 3).tolist()
                            output_frame = self.apply_visualizations(
                                output_frame,
                                processed_frame,
                                box_coords,
                                masks[0],
                                color
                            )
                            logger.debug(f"Successfully processed person {i+1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing person {i+1}: {str(e)}")
                        continue
                
                self.last_processed_frame = output_frame
            
            # Return the last processed frame or the original frame if we haven't processed any yet
            return self.last_processed_frame if self.last_processed_frame is not None else original_frame
            
        except Exception as e:
            logger.error(f"Error in process_frame: {str(e)}")
            return self.last_processed_frame if self.last_processed_frame is not None else None
    
    def run(self, output_window_name='Human Segmentation'):
        """Continuously process and display frames"""
        try:
            while True:
                frame = self.process_frame()
                
                if frame is None:
                    logger.error("Stream ended or error occurred")
                    break
                
                # Display frame
                cv2.imshow(output_window_name, frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
        
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function to run the RTSP Human Segmentation pipeline"""
    # Build RTSP URL from environment variables
    try:
        rtsp_url = f"rtsp://{os.getenv('RTSP_USERNAME')}:{os.getenv('RTSP_PASSWORD')}@{os.getenv('RTSP_IP')}:{os.getenv('RTSP_PORT')}/Streaming/Channels/{os.getenv('RTSP_CHANNEL')}"
    except Exception as e:
        logger.error(f"Error building RTSP URL. Please check your .env file: {str(e)}")
        return
    
    # Initialize and run segmentation
    try:
        segmenter = RTSPHumanSegmenter(
            rtsp_url, 
            process_every_n_frames=30, 
            process_width=640, 
            process_height=480,
            confidence_threshold=0.7  # Only detect humans with 70% or higher confidence
        )
        segmenter.run()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main()

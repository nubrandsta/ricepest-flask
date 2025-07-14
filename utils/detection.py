import time
import numpy as np
from PIL import Image

# Try to import cv2, fallback to PIL if it fails
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV not available: {e}. Using PIL fallback.")
    CV2_AVAILABLE = False

# Try to import ultralytics and related packages
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    print(f"Ultralytics not available: {e}. YOLO detection will be disabled.")
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
    SAHI_AVAILABLE = True
except ImportError as e:
    print(f"SAHI not available: {e}. SAHI detection will be disabled.")
    SAHI_AVAILABLE = False
    AutoDetectionModel = None
    get_sliced_prediction = None
    read_image = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch not available: {e}. GPU acceleration will be disabled.")
    TORCH_AVAILABLE = False
    torch = None

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        float: IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for considering boxes as overlapping
    
    Returns:
        List of filtered detections
    """
    if not detections:
        return detections
    
    # Sort detections by confidence score (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    removed_count = 0
    
    for current_det in sorted_detections:
        should_keep = True
        
        for kept_det in filtered_detections:
            # Only compare detections of the same class
            if current_det['class'] == kept_det['class']:
                iou = calculate_iou(current_det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    removed_count += 1
                    print(f"Removing overlapping {current_det['class']} detection (IoU: {iou:.3f} > {iou_threshold})")
                    break
        
        if should_keep:
            filtered_detections.append(current_det)
    
    print(f"NMS removed {removed_count} overlapping detections")
    return filtered_detections

class YOLODetector:
    def __init__(self, model_path='model/yolov11s.pt'):
        """
        Initialize YOLOv11 detector with SAHI support
        
        Args:
            model_path (str): Path to the YOLOv11 model file
        """
        self.model_path = model_path
        self.model = None
        self.sahi_model = None
        self.available = ULTRALYTICS_AVAILABLE
        
        if not ULTRALYTICS_AVAILABLE:
            print("Warning: Ultralytics not available. YOLODetector will not function.")
            return
            
        self.load_model()
    
    def load_model(self):
        """
        Load YOLOv11 model for both standard and SAHI detection
        """
        try:
            # Load standard YOLO model
            self.model = YOLO(self.model_path)
            
            # Initialize SAHI model wrapper if SAHI is available
            if SAHI_AVAILABLE:
                device = 'cpu'  # Default to CPU
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    device = 'cuda'
                    
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',  # SAHI uses yolov8 type for YOLOv11 compatibility
                    model_path=self.model_path,
                    confidence_threshold=0.3,
                    device=device
                )
            else:
                print("Warning: SAHI not available. SAHI detection will be disabled.")
                self.sahi_model = None
            
            print(f"Model loaded successfully from {self.model_path}")
            if TORCH_AVAILABLE:
                print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            else:
                print("Using device: CPU (PyTorch not available)")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def detect(self, image_path, use_sahi=False, sahi_config=None):
        """
        Run detection on image with optional SAHI slicing
        
        Args:
            image_path (str): Path to input image
            use_sahi (bool): Whether to use SAHI for detection
            sahi_config (dict): SAHI configuration parameters
            
        Returns:
            dict: Detection results with bounding boxes, confidence scores, and metadata
        """
        start_time = time.time()
        
        # Check if detector is available
        if not self.available or self.model is None:
            return {
                'detections': [],
                'processing_time': time.time() - start_time,
                'image_size': [0, 0],
                'method': 'unavailable',
                'error': 'YOLO detector not available due to missing dependencies'
            }
        
        # Check SAHI availability if requested
        if use_sahi and not SAHI_AVAILABLE:
            print("Warning: SAHI requested but not available. Falling back to standard detection.")
            use_sahi = False
        
        if use_sahi:
            return self._detect_with_sahi(image_path, sahi_config, start_time)
        else:
            return self._detect_standard(image_path, start_time)
    
    def _detect_standard(self, image_path, start_time):
        """
        Standard YOLOv11 detection without slicing
        """
        # Load image - use PIL fallback if OpenCV not available
        if CV2_AVAILABLE:
            image = cv2.imread(image_path)
            image_size = [image.shape[1], image.shape[0]]
        else:
            # Use PIL as fallback
            pil_image = Image.open(image_path)
            image_size = [pil_image.width, pil_image.height]
        
        # Run inference (YOLO can handle the image path directly)
        results = self.model(image_path, conf=0.3)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        if TORCH_AVAILABLE:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                        else:
                            # Fallback for when torch is not available
                            x1, y1, x2, y2 = box.xyxy[0].numpy()
                            confidence = box.conf[0].numpy()
                            class_id = int(box.cls[0].numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id]
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        })
                    except Exception as e:
                        print(f"Error processing detection box: {e}")
                        continue
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_size': image_size,
            'method': 'standard'
        }
    
    def _detect_with_sahi(self, image_path, sahi_config, start_time):
        """
        SAHI-based detection with image slicing
        """
        # Check if SAHI is available
        if not SAHI_AVAILABLE or self.sahi_model is None:
            print("Error: SAHI not available for sliced detection")
            return {
                'detections': [],
                'processing_time': time.time() - start_time,
                'image_size': [0, 0],
                'method': 'sahi_unavailable',
                'error': 'SAHI not available for sliced detection'
            }
        
        # Default SAHI configuration with more aggressive NMS
        default_config = {
            'slice_height': 640,
            'slice_width': 640,
            'overlap_height_ratio': 0.2,
            'overlap_width_ratio': 0.2,
            'nms_threshold': 0.3  # More aggressive NMS for SAHI
        }
        
        if sahi_config:
            default_config.update(sahi_config)
        
        # Load image
        image = read_image(image_path)
        
        # Run SAHI prediction
        result = get_sliced_prediction(
            image,
            self.sahi_model,
            slice_height=default_config['slice_height'],
            slice_width=default_config['slice_width'],
            overlap_height_ratio=default_config['overlap_height_ratio'],
            overlap_width_ratio=default_config['overlap_width_ratio']
        )
        
        # Process SAHI results
        detections = []
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox
            
            detections.append({
                'bbox': [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)],
                'confidence': float(object_prediction.score.value),
                'class': object_prediction.category.name,
                'class_id': object_prediction.category.id
            })
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        print(f"Before NMS: {len(detections)} detections")
        detections = apply_nms(detections, iou_threshold=default_config['nms_threshold'])
        print(f"After NMS: {len(detections)} detections (threshold: {default_config['nms_threshold']})")
        
        # Debug: Print detection details
        for i, det in enumerate(detections):
            print(f"Detection {i+1}: {det['class']} - confidence: {det['confidence']:.3f} - bbox: {det['bbox']}")
        
        processing_time = time.time() - start_time
        
        # Calculate number of slices
        image_height, image_width = image.shape[:2]
        slices_y = int(np.ceil(image_height / (default_config['slice_height'] * (1 - default_config['overlap_height_ratio']))))
        slices_x = int(np.ceil(image_width / (default_config['slice_width'] * (1 - default_config['overlap_width_ratio']))))
        slice_count = slices_x * slices_y
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_size': [image_width, image_height],
            'slice_count': slice_count,
            'method': 'sahi',
            'sahi_config': default_config
        }
    
    def get_version(self):
        """
        Get Ultralytics version
        """
        try:
            import ultralytics
            return ultralytics.__version__
        except:
            return "Unknown"
    
    def get_model_info(self):
        """
        Get model information
        """
        if self.model is None:
            return None
            
        return {
            'model_path': self.model_path,
            'model_type': 'YOLOv11',
            'classes': list(self.model.names.values()) if self.model.names else [],
            'device': str(next(self.model.model.parameters()).device) if self.model.model else 'unknown'
        }
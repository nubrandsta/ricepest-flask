import time
import os
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

def boxes_intersect(box1, box2):
    """
    Check if two bounding boxes intersect.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        bool: True if boxes intersect, False otherwise
    """
    return not (box1[2] <= box2[0] or  # box1 is to the left of box2
                box2[2] <= box1[0] or  # box2 is to the left of box1
                box1[3] <= box2[1] or  # box1 is above box2
                box2[3] <= box1[1])    # box2 is above box1

def filter_pest_detections_by_nonpest(pest_detections, nonpest_detections):
    """
    Filter out pest detections that intersect with nonpest detections.
    
    Args:
        pest_detections: List of pest detection dictionaries
        nonpest_detections: List of nonpest detection dictionaries
    
    Returns:
        List of filtered pest detections
    """
    if not nonpest_detections:
        return pest_detections
    
    filtered_detections = []
    removed_count = 0
    
    for pest_det in pest_detections:
        should_keep = True
        
        for nonpest_det in nonpest_detections:
            if boxes_intersect(pest_det['bbox'], nonpest_det['bbox']):
                should_keep = False
                removed_count += 1
                print(f"Removing pest detection '{pest_det['class']}' (conf: {pest_det['confidence']:.3f}) that intersects with nonpest detection '{nonpest_det['class']}'")
                break
        
        if should_keep:
            filtered_detections.append(pest_det)
    
    print(f"Two-stage filtering removed {removed_count} pest detections that intersected with nonpest areas")
    return filtered_detections

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
    def __init__(self, model_path='model/yolov11s.pt', nonpest_model_path='model/nonpest.pt'):
        """
        Initialize YOLOv11 detector with SAHI support and two-stage detection
        
        Args:
            model_path (str): Path to the main pest detection YOLOv11 model file
            nonpest_model_path (str): Path to the nonpest detection model file
        """
        self.model_path = model_path
        self.nonpest_model_path = nonpest_model_path
        self.model = None
        self.nonpest_model = None
        self.sahi_model = None
        self.available = ULTRALYTICS_AVAILABLE
        
        if not ULTRALYTICS_AVAILABLE:
            print("Warning: Ultralytics not available. YOLODetector will not function.")
            return
            
        self.load_model()
    
    def load_model(self):
        """
        Load YOLOv11 models for both pest detection and nonpest filtering
        """
        try:
            # Load main pest detection model
            self.model = YOLO(self.model_path)
            print(f"Pest detection model loaded successfully from {self.model_path}")
            
            # Load nonpest detection model
            if self.nonpest_model_path is not None and os.path.exists(self.nonpest_model_path):
                self.nonpest_model = YOLO(self.nonpest_model_path)
                print(f"Nonpest detection model loaded successfully from {self.nonpest_model_path}")
            else:
                if self.nonpest_model_path is None:
                    print("Warning: No nonpest model path provided. Two-stage detection will be disabled.")
                else:
                    print(f"Warning: Nonpest model not found at {self.nonpest_model_path}. Two-stage detection will be disabled.")
                self.nonpest_model = None
            
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
            
            if TORCH_AVAILABLE:
                print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            else:
                print("Using device: CPU (PyTorch not available)")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def detect(self, image_path, use_sahi=False, sahi_config=None, use_two_stage=True):
        """
        Run detection on image with optional SAHI slicing and two-stage filtering
        
        Args:
            image_path (str): Path to input image
            use_sahi (bool): Whether to use SAHI for detection
            sahi_config (dict): SAHI configuration parameters
            use_two_stage (bool): Whether to use two-stage detection with nonpest filtering
            
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
            return self._detect_with_sahi(image_path, sahi_config, start_time, use_two_stage)
        else:
            return self._detect_standard(image_path, start_time, use_two_stage)
    
    def _detect_nonpest(self, image_path):
        """
        Run nonpest detection to identify ads, banners, and other non-pest objects
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            list: List of nonpest detections
        """
        if self.nonpest_model is None:
            return []
        
        try:
            # Run nonpest detection
            results = self.nonpest_model(image_path, conf=0.3)
            
            nonpest_detections = []
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
                            
                            # Get class name from nonpest model, but label all as 'lain'
                            original_class_name = self.nonpest_model.names[class_id]
                            
                            nonpest_detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence),
                                'class': 'lain',  # Label all nonpest detections as 'lain'
                                'class_id': class_id,
                                'original_class': original_class_name
                            })
                        except Exception as e:
                            print(f"Error processing nonpest detection box: {e}")
                            continue
            
            print(f"Nonpest model detected {len(nonpest_detections)} non-pest objects")
            return nonpest_detections
            
        except Exception as e:
            print(f"Error running nonpest detection: {e}")
            return []
    
    def _detect_standard(self, image_path, start_time, use_two_stage=True):
        """
        Standard YOLOv11 detection without slicing, with optional two-stage filtering
        """
        # Load image - use PIL fallback if OpenCV not available
        if CV2_AVAILABLE:
            image = cv2.imread(image_path)
            image_size = [image.shape[1], image.shape[0]]
        else:
            # Use PIL as fallback
            pil_image = Image.open(image_path)
            image_size = [pil_image.width, pil_image.height]
        
        # Stage 1: Run nonpest detection if two-stage is enabled
        nonpest_detections = []
        if use_two_stage and self.nonpest_model is not None:
            print("Running Stage 1: Nonpest detection...")
            nonpest_detections = self._detect_nonpest(image_path)
        
        # Stage 2: Run main pest detection
        print("Running Stage 2: Pest detection...")
        results = self.model(image_path, conf=0.3)
        
        # Process pest detection results
        pest_detections = []
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
                        
                        pest_detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class': class_name,
                            'class_id': class_id
                        })
                    except Exception as e:
                        print(f"Error processing detection box: {e}")
                        continue
        
        print(f"Before filtering: {len(pest_detections)} pest detections, {len(nonpest_detections)} nonpest detections")
        
        # Apply two-stage filtering if enabled
        if use_two_stage and nonpest_detections:
            pest_detections = filter_pest_detections_by_nonpest(pest_detections, nonpest_detections)
        
        # Combine all detections for visualization (pest + nonpest)
        all_detections = pest_detections + nonpest_detections
        
        processing_time = time.time() - start_time
        
        return {
            'detections': all_detections,
            'pest_detections': pest_detections,
            'nonpest_detections': nonpest_detections,
            'processing_time': processing_time,
            'image_size': image_size,
            'method': 'standard_two_stage' if use_two_stage else 'standard'
        }
    
    def _detect_with_sahi(self, image_path, sahi_config, start_time, use_two_stage=True):
        """
        SAHI-based detection with image slicing and optional two-stage filtering
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
        
        # Stage 1: Run nonpest detection if two-stage is enabled
        nonpest_detections = []
        if use_two_stage and self.nonpest_model is not None:
            print("Running Stage 1: Nonpest detection...")
            nonpest_detections = self._detect_nonpest(image_path)
        
        # Stage 2: Run main pest detection with SAHI
        print("Running Stage 2: SAHI pest detection...")
        
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
        pest_detections = []
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox
            
            pest_detections.append({
                'bbox': [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)],
                'confidence': float(object_prediction.score.value),
                'class': object_prediction.category.name,
                'class_id': object_prediction.category.id
            })
        
        # Apply Non-Maximum Suppression to remove overlapping detections
        print(f"Before NMS: {len(pest_detections)} detections")
        pest_detections = apply_nms(pest_detections, iou_threshold=default_config['nms_threshold'])
        print(f"After NMS: {len(pest_detections)} detections (threshold: {default_config['nms_threshold']})")
        
        print(f"Before filtering: {len(pest_detections)} pest detections, {len(nonpest_detections)} nonpest detections")
        
        # Apply two-stage filtering if enabled
        if use_two_stage and nonpest_detections:
            pest_detections = filter_pest_detections_by_nonpest(pest_detections, nonpest_detections)
        
        # Combine all detections for visualization (pest + nonpest)
        all_detections = pest_detections + nonpest_detections
        
        # Debug: Print detection details
        for i, det in enumerate(all_detections):
            print(f"Detection {i+1}: {det['class']} - confidence: {det['confidence']:.3f} - bbox: {det['bbox']}")
        
        processing_time = time.time() - start_time
        
        # Calculate number of slices
        image_height, image_width = image.shape[:2]
        slices_y = int(np.ceil(image_height / (default_config['slice_height'] * (1 - default_config['overlap_height_ratio']))))
        slices_x = int(np.ceil(image_width / (default_config['slice_width'] * (1 - default_config['overlap_width_ratio']))))
        slice_count = slices_x * slices_y
        
        return {
            'detections': all_detections,
            'pest_detections': pest_detections,
            'nonpest_detections': nonpest_detections,
            'processing_time': processing_time,
            'image_size': [image_width, image_height],
            'slice_count': slice_count,
            'method': 'sahi_two_stage' if use_two_stage else 'sahi',
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
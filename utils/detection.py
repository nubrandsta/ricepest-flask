import time
import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from PIL import Image
import torch

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
        self.load_model()
    
    def load_model(self):
        """
        Load YOLOv11 model for both standard and SAHI detection
        """
        try:
            # Load standard YOLO model
            self.model = YOLO(self.model_path)
            
            # Initialize SAHI model wrapper
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',  # SAHI uses yolov8 type for YOLOv11 compatibility
                model_path=self.model_path,
                confidence_threshold=0.3,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            print(f"Model loaded successfully from {self.model_path}")
            print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
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
        
        if use_sahi:
            return self._detect_with_sahi(image_path, sahi_config, start_time)
        else:
            return self._detect_standard(image_path, start_time)
    
    def _detect_standard(self, image_path, start_time):
        """
        Standard YOLOv11 detection without slicing
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(image_path, conf=0.3)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class': class_name,
                        'class_id': class_id
                    })
        
        processing_time = time.time() - start_time
        
        return {
            'detections': detections,
            'processing_time': processing_time,
            'image_size': [image.shape[1], image.shape[0]],
            'method': 'standard'
        }
    
    def _detect_with_sahi(self, image_path, sahi_config, start_time):
        """
        SAHI-based detection with image slicing
        """
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
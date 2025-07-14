# Two-Stage Detection System

This document describes the implementation of the two-stage detection algorithm that helps eliminate false positives by filtering out pest detections that intersect with non-pest objects like ads, banners, and other irrelevant content.

## Overview

The two-stage detection system works as follows:

1. **Stage 1: Non-pest Detection** - A specialized model detects ads, banners, and other non-pest objects
2. **Stage 2: Pest Detection** - The main pest detection model identifies potential pests
3. **Filtering** - Any pest detections that intersect with non-pest areas are removed
4. **Visualization** - Non-pest detections are labeled as 'lain' for display

## Implementation Details

### Key Components

1. **YOLODetector Class Enhancement**
   - Added `nonpest_model_path` parameter to constructor
   - Loads both main pest model and non-pest model
   - Supports optional two-stage detection

2. **New Functions**
   - `boxes_intersect()` - Checks if two bounding boxes overlap
   - `filter_pest_detections_by_nonpest()` - Removes intersecting pest detections
   - `_detect_nonpest()` - Runs non-pest detection

3. **Updated Detection Methods**
   - `_detect_standard()` - Enhanced with two-stage support
   - `_detect_with_sahi()` - Enhanced with two-stage support
   - `detect()` - Added `use_two_stage` parameter

### API Changes

#### Initialization
```python
# Initialize with both models
detector = YOLODetector(
    model_path='model/yolov11s.pt',
    nonpest_model_path='model/nonpest_model.pt'
)
```

#### Detection
```python
# Run two-stage detection (default)
results = detector.detect(image_path, use_two_stage=True)

# Run standard detection only
results = detector.detect(image_path, use_two_stage=False)
```

#### API Endpoints

Both `/api/detect` and `/api/detect-sahi` endpoints now support:

**Request Parameters:**
- `use_two_stage` (boolean, default: True) - Enable/disable two-stage detection

**Response Fields:**
- `detections` - All detections (pest + nonpest)
- `pest_detections` - Only pest detections (after filtering)
- `nonpest_detections` - Only nonpest detections (labeled as 'lain')
- `two_stage_enabled` - Whether two-stage detection was used

### Example Usage

#### Python Script
```python
from utils.detection import YOLODetector

# Initialize detector
detector = YOLODetector(
    model_path='model/yolov11s.pt',
    nonpest_model_path='model/nonpest_model.pt'
)

# Run detection
results = detector.detect('image.jpg', use_two_stage=True)

print(f"Total detections: {len(results['detections'])}")
print(f"Pest detections: {len(results['pest_detections'])}")
print(f"Nonpest detections: {len(results['nonpest_detections'])}")
```

#### API Request
```javascript
fetch('/api/detect', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        image: base64_image_data,
        use_two_stage: true
    })
})
.then(response => response.json())
.then(data => {
    console.log('All detections:', data.detections);
    console.log('Pest detections:', data.pest_detections);
    console.log('Nonpest detections:', data.nonpest_detections);
    console.log('Two-stage enabled:', data.two_stage_enabled);
});
```

## Model Requirements

### Main Pest Detection Model
- Path: `model/yolov11s.pt`
- Purpose: Detect various pest species
- Required: Yes

### Non-pest Detection Model
- Path: `model/nonpest_model.pt`
- Purpose: Detect ads, banners, and other non-pest objects
- Classes: Should have 2 classes representing different types of non-pest content
- Required: No (system falls back to single-stage detection if not available)

## Configuration

### Model Paths
Update the model paths in `app.py`:
```python
nonpest_model_path = 'model/nonpest_model.pt'  # Update this path
detector = YOLODetector(
    model_path='model/yolov11s.pt',
    nonpest_model_path=nonpest_model_path if os.path.exists(nonpest_model_path) else None
)
```

### Detection Confidence
Both models use a confidence threshold of 0.3 by default. This can be adjusted in the detection methods:
```python
results = self.model(image_path, conf=0.3)  # Adjust confidence threshold
```

## Health Check

The `/api/health` endpoint now includes information about two-stage detection:
```json
{
    "status": "healthy",
    "detector_available": true,
    "model_loaded": true,
    "nonpest_model_loaded": true,
    "two_stage_available": true,
    "dependencies": {
        "ultralytics": true,
        "sahi": true,
        "torch": true,
        "opencv": true
    }
}
```

## Testing

Use the provided example script to test the two-stage detection:
```bash
python example_two_stage_detection.py path/to/test/image.jpg
```

This script will:
1. Run standard detection without filtering
2. Run two-stage detection with filtering
3. Compare the results and show what was filtered out

## Benefits

1. **Reduced False Positives** - Eliminates pest detections in ad/banner areas
2. **Improved Accuracy** - More reliable pest detection results
3. **Flexible** - Can be enabled/disabled per request
4. **Backward Compatible** - Existing API calls work unchanged
5. **Informative** - Provides detailed breakdown of detection types

## Troubleshooting

### Common Issues

1. **Nonpest model not loading**
   - Check if `model/nonpest_model.pt` exists
   - Verify the model file is not corrupted
   - Check console output for loading errors

2. **Two-stage detection not working**
   - Verify `nonpest_model_loaded` is `true` in health check
   - Ensure `use_two_stage` parameter is set to `true`
   - Check console output for detection logs

3. **Performance issues**
   - Two-stage detection takes longer as it runs two models
   - Consider using SAHI with two-stage for better performance on large images
   - Monitor processing times in the response

### Debug Information

The system provides detailed console output during detection:
```
Running Stage 1: Nonpest detection...
Nonpest model detected 2 non-pest objects
Running Stage 2: Pest detection...
Before filtering: 5 pest detections, 2 nonpest detections
Removing pest detection 'aphid' (conf: 0.654) that intersects with nonpest detection 'lain'
Two-stage filtering removed 1 pest detections that intersected with nonpest areas
```
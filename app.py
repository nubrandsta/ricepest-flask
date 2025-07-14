from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import uuid
from utils.detection import YOLODetector

app = Flask(__name__)
CORS(app)

# Initialize detector (loads once at startup)
try:
    # Initialize with main pest detection model and optional nonpest model
    nonpest_model_path = 'model/nonpest.pt'  # Path to your nonpest detection model
    detector = YOLODetector(
        model_path='model/yolov11s.pt',
        nonpest_model_path=nonpest_model_path if os.path.exists(nonpest_model_path) else None
    )
    detector_available = detector.available
    print(f"Detector initialized. Two-stage detection available: {detector.nonpest_model is not None}")
except Exception as e:
    print(f"Failed to initialize YOLODetector: {e}")
    detector = None
    detector_available = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_image():
    try:
        # Check if detector is available
        if not detector_available:
            return jsonify({
                'error': 'YOLO detector not available due to missing dependencies (ultralytics, torch, etc.)',
                'detections': [],
                'processing_time': 0,
                'image_size': [0, 0],
                'method': 'unavailable'
            }), 503
        
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporary image
        temp_id = str(uuid.uuid4())
        temp_path = f'static/uploads/{temp_id}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        image.save(temp_path)
        
        # Get two-stage detection parameter (default: True)
        use_two_stage = data.get('use_two_stage', True)
        
        # Run standard detection
        results = detector.detect(temp_path, use_sahi=False, use_two_stage=use_two_stage)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'detections': results['detections'],
            'pest_detections': results.get('pest_detections', []),
            'nonpest_detections': results.get('nonpest_detections', []),
            'processing_time': results['processing_time'],
            'image_size': results['image_size'],
            'method': results.get('method', 'standard'),
            'two_stage_enabled': use_two_stage and detector.nonpest_model is not None,
            'error': results.get('error')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-sahi', methods=['POST'])
def detect_with_sahi():
    try:
        # Check if detector is available
        if not detector_available:
            return jsonify({
                'error': 'YOLO detector not available due to missing dependencies (ultralytics, torch, sahi, etc.)',
                'detections': [],
                'processing_time': 0,
                'image_size': [0, 0],
                'slice_count': 0,
                'method': 'unavailable'
            }), 503
        
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporary image
        temp_id = str(uuid.uuid4())
        temp_path = f'static/uploads/{temp_id}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        image.save(temp_path)
        
        # Get SAHI config and two-stage detection parameter
        sahi_config = data.get('sahi_config', {})
        use_two_stage = data.get('use_two_stage', True)
        
        # Run SAHI detection
        results = detector.detect(temp_path, use_sahi=True, sahi_config=sahi_config, use_two_stage=use_two_stage)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'detections': results['detections'],
            'pest_detections': results.get('pest_detections', []),
            'nonpest_detections': results.get('nonpest_detections', []),
            'processing_time': results['processing_time'],
            'image_size': results['image_size'],
            'slice_count': results.get('slice_count', 0),
            'method': results.get('method', 'sahi'),
            'two_stage_enabled': use_two_stage and detector.nonpest_model is not None,
            'error': results.get('error')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    # Check dependency availability
    try:
        from utils.detection import ULTRALYTICS_AVAILABLE, SAHI_AVAILABLE, TORCH_AVAILABLE, CV2_AVAILABLE
    except ImportError:
        ULTRALYTICS_AVAILABLE = SAHI_AVAILABLE = TORCH_AVAILABLE = CV2_AVAILABLE = False
    
    health_status = {
        'status': 'healthy' if detector_available else 'degraded',
        'detector_available': detector_available,
        'model_loaded': detector.model is not None if detector else False,
        'nonpest_model_loaded': detector.nonpest_model is not None if detector else False,
        'two_stage_available': detector.nonpest_model is not None if detector else False,
        'dependencies': {
            'ultralytics': ULTRALYTICS_AVAILABLE,
            'sahi': SAHI_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'opencv': CV2_AVAILABLE
        }
    }
    
    # Add version info if available
    if detector_available and detector:
        try:
            health_status['ultralytics_version'] = detector.get_version()
            health_status['model_info'] = detector.get_model_info()
        except:
            health_status['ultralytics_version'] = 'Unknown'
            health_status['model_info'] = None
    
    return jsonify(health_status)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
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
detector = YOLODetector(model_path='model/yolov11s.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_image():
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporary image
        temp_id = str(uuid.uuid4())
        temp_path = f'static/uploads/{temp_id}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        image.save(temp_path)
        
        # Run standard detection
        results = detector.detect(temp_path, use_sahi=False)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'detections': results['detections'],
            'processing_time': results['processing_time'],
            'image_size': results['image_size'],
            'method': 'standard'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-sahi', methods=['POST'])
def detect_with_sahi():
    try:
        data = request.get_json()
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Save temporary image
        temp_id = str(uuid.uuid4())
        temp_path = f'static/uploads/{temp_id}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        image.save(temp_path)
        
        # Get SAHI config
        sahi_config = data.get('sahi_config', {})
        
        # Run SAHI detection
        results = detector.detect(temp_path, use_sahi=True, sahi_config=sahi_config)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'detections': results['detections'],
            'processing_time': results['processing_time'],
            'image_size': results['image_size'],
            'slice_count': results.get('slice_count', 0),
            'method': 'sahi'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'ultralytics_version': detector.get_version()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
# Rice Pest Detection - Deployment Guide

## Railway Deployment Configuration

This application has been configured for Railway deployment with proper handling of ultralytics dependencies. The configuration includes:

### 1. Railway Configuration Files

#### `railway.toml`
- Configures Nixpacks builder with Python 3.11
- Installs system dependencies for OpenCV and ultralytics
- Custom install command for proper PyTorch and ultralytics installation
- Environment variables for headless operation

#### `nixpacks.toml`
- Additional Nixpacks configuration
- Comprehensive system package list
- Staged installation process for dependencies
- Optimized for CPU-only PyTorch deployment

### 2. Dependency Management

#### Modified `requirements.txt`
- Core Flask dependencies
- ultralytics, torch, torchvision, and sahi are installed via custom commands
- Prevents dependency conflicts during build

#### Custom Install Process
1. System packages installation
2. Python package upgrades
3. PyTorch CPU installation from official index
4. ultralytics and sahi installation
5. Remaining requirements installation

### 3. Error Handling

The application includes comprehensive fallback mechanisms:
- Graceful handling of missing ultralytics
- SAHI availability checks
- OpenCV fallback to PIL
- Informative error responses
- Health check endpoint with dependency status

## Alternative Deployment Options

### Option 1: Railway with Custom Build Commands

In your Railway dashboard, you can override the build process:

#### Updated Custom Build Commands (Ubuntu Noble Compatible):

**Full feature build command:**
```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 libavcodec-dev libavformat-dev libswscale-dev libfontconfig1 libcairo2 libgdk-pixbuf2.0-0 libpango-1.0-0 libharfbuzz0b libpangocairo-1.0-0 libatk1.0-0 libcairo-gobject2 libjpeg-dev libpng-dev libtiff-dev libwebp-dev libopenjp2-7-dev && pip install --upgrade pip setuptools wheel && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install ultralytics>=8.0.0 sahi==0.11.14 && pip install -r requirements.txt
```

**Minimal build command** (if above fails):
```bash
apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libfontconfig1 libcairo2 libjpeg-dev libpng-dev && pip install --upgrade pip setuptools wheel && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install ultralytics>=8.0.0 sahi==0.11.14 && pip install -r requirements.txt
```

**Custom Deploy Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

#### Package Compatibility Notes:
- Ubuntu Noble (24.04) uses different package names for FFmpeg libraries
- Changed `libavcodec58` → `libavcodec-dev`
- Changed `libavformat58` → `libavformat-dev` 
- Changed `libswscale5` → `libswscale-dev`
- Use `railway-minimal.toml` if you encounter persistent package issues

### Option 2: Docker Deployment

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install ultralytics>=8.0.0 sahi==0.11.14 && \
    pip install -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV ULTRALYTICS_SETTINGS='{"runs_dir": "/tmp", "datasets_dir": "/tmp", "weights_dir": "/tmp"}'

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Option 3: Heroku Deployment

For Heroku, create these files:

**`Aptfile`:**
```
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
```

**`runtime.txt`:**
```
python-3.11.0
```

**Buildpacks:**
1. `heroku/python`
2. `heroku-community/apt`

### Option 4: Google Cloud Run

Use the Docker configuration above with Cloud Run:

```bash
# Build and push to Google Container Registry
docker build -t gcr.io/YOUR_PROJECT_ID/ricepest-detection .
docker push gcr.io/YOUR_PROJECT_ID/ricepest-detection

# Deploy to Cloud Run
gcloud run deploy ricepest-detection \
  --image gcr.io/YOUR_PROJECT_ID/ricepest-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 5: AWS Lambda (Serverless)

For serverless deployment, consider using:
- AWS Lambda with container images
- Serverless Framework
- Zappa for Python applications

**Note:** Lambda has size limitations that may require model optimization.

## Troubleshooting

### Common Issues

1. **Package Installation Errors (Ubuntu Noble/24.04)**
   - **Problem**: `libavcodec58`, `libavformat58`, `libswscale5` not available
   - **Solution**: Use development packages instead: `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`
   - **Alternative**: Use `railway-minimal.toml` for basic functionality

2. **ultralytics Import Error**
   - Ensure PyTorch is installed before ultralytics
   - Use CPU-only PyTorch for most cloud deployments
   - Check system dependencies are installed
   - Verify package versions in build logs

3. **Memory Issues**
   - Use smaller YOLO models (yolov8n.pt instead of yolov8x.pt)
   - Implement model caching
   - Consider serverless cold start times

4. **OpenCV Issues**
   - Use opencv-python-headless for server deployments
   - Install required system libraries
   - Set QT_QPA_PLATFORM=offscreen

5. **Railway Build Failures**
   - Check build logs for specific package errors
   - Try the minimal configuration: `railway-minimal.toml`
   - Use custom build commands in Railway dashboard
   - Verify Ubuntu version compatibility

### Health Check

Use the `/api/health` endpoint to verify:
- Dependency availability
- Model loading status
- System configuration

### Environment Variables

Set these for optimal performance:
```bash
QT_QPA_PLATFORM=offscreen
DISPLAY=:99
OPENCV_IO_ENABLE_OPENEXR=1
ULTRALYTICS_SETTINGS='{"runs_dir": "/tmp", "datasets_dir": "/tmp", "weights_dir": "/tmp"}'
```

## Performance Optimization

1. **Model Selection**
   - Use YOLOv8n for faster inference
   - Consider model quantization
   - Implement model caching

2. **SAHI Configuration**
   - Adjust slice size based on image resolution
   - Optimize overlap ratio
   - Use appropriate confidence thresholds

3. **Resource Management**
   - Implement request queuing for high load
   - Use connection pooling
   - Monitor memory usage

This configuration ensures reliable deployment of the ultralytics-based rice pest detection system across various cloud platforms.
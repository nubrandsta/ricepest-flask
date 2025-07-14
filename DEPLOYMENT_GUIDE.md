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

## Deployment Options

### Option 1: Railway with Railpack (Recommended)

Railpack is a modern build system with superior Python support compared to Nixpacks.

#### Why Railpack?
- **Automatic Python Detection**: Detects Python projects via requirements.txt, main.py, or pyproject.toml
- **Built-in Dependency Management**: Automatically installs system dependencies for common packages
- **Multiple Package Manager Support**: Works with pip, poetry, pdm, uv, and pipenv
- **Optimized Runtime Variables**: Includes production-ready Python environment settings

#### Configuration Files:

1. **`railpack.toml`** - Modern Railpack configuration
2. **`railway.toml`** - Railway settings for Railpack builder (active)

### Option 2: Railway with Nixpacks (Legacy)

Nixpacks configuration for compatibility with older deployments.

#### Configuration Files:

1. **`railway-nixpacks-backup.toml`** - Backup of legacy Nixpacks Railway settings
2. **`runtime.txt`** - Python version specification

**Note**: Legacy Nixpacks files have been removed. Use the backup file if you need to revert to Nixpacks.

### Option 3: Railway with Custom Build Commands

In your Railway dashboard, you can override the build process:

#### Updated Custom Build Commands (Ubuntu Noble Compatible):

**Full feature build command:**
```bash
python -m pip install --upgrade pip setuptools wheel && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && python -m pip install ultralytics sahi && python -m pip install -r requirements.txt
```

**Minimal build command** (if above fails):
```bash
python -m pip install --upgrade pip setuptools wheel && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && python -m pip install ultralytics>=8.0.0 sahi==0.11.14 && python -m pip install -r requirements.txt
```

**Note**: System dependencies are now handled automatically by `nixpacks.toml` configuration.

**Custom Deploy Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

#### Package Compatibility Notes:
- Ubuntu Noble (24.04) uses different package names for system libraries
- **OpenGL**: Changed `libgl1-mesa-glx` → `libgl1-mesa-dri` (obsolete package)
- **FFmpeg**: Changed `libavcodec58` → `libavcodec-dev`
- **FFmpeg**: Changed `libavformat58` → `libavformat-dev` 
- **FFmpeg**: Changed `libswscale5` → `libswscale-dev`
- Use `railway-minimal.toml` if you encounter persistent package issues

### Option 4: Docker Deployment

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

### Option 5: Heroku Deployment

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

### Option 6: Google Cloud Run

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

### Option 7: AWS Lambda (Serverless)

For serverless deployment, consider using:
- AWS Lambda with container images
- Serverless Framework
- Zappa for Python applications

**Note:** Lambda has size limitations that may require model optimization.

## Troubleshooting

### Using Railpack (Current Setup)

The project is now configured to use Railpack by default:

**Benefits of Current Setup:**
- Automatic Python environment setup with proper pip integration
- Built-in support for common Python packages (no manual system dependency management)
- Faster builds with optimized caching
- Better error messages and debugging information

**Current Configuration:**
1. `railway.toml` - Active Railpack configuration
2. `railpack.toml` - Railpack-specific settings
3. Legacy Nixpacks files have been removed for simplicity

### Common Issues (Nixpacks Legacy)

1. **Package Installation Errors (Ubuntu Noble/24.04)**
   - **Problem**: `libavcodec58`, `libavformat58`, `libswscale5` not available
   - **Solution**: Use development packages instead: `libavcodec-dev`, `libavformat-dev`, `libswscale-dev`
   - **Alternative**: Use `railway-minimal.toml` for basic functionality
   - **Recommended**: Migrate to Railpack which handles this automatically

2. **Pip Command Not Found**
   - **Problem**: `pip: command not found` during deployment
   - **Solution**: Use `python -m pip` instead of `pip` directly
   - **Reason**: This ensures pip is called through the Python module system
   - **Note**: All configuration files have been updated to use `python -m pip`
   - **Recommended**: Migrate to Railpack which handles pip automatically

3. **Python/Pip Not Available (Nixpacks)**
   - **Problem**: `python: command not found` or `No module named pip`
   - **Solution**: Use `nixPkgs = ["python3Full"]` and add pip installation fallback in setup phase
   - **Fallback**: `cmds = ["python -m ensurepip --upgrade || curl -sS https://bootstrap.pypa.io/get-pip.py | python"]`
   - **Cause**: Nix environment may not include pip even with python3Packages.pip
   - **Note**: python3Full includes more complete Python environment with pip support

4. **Nixpacks Configuration Parse Error**
   - **Problem**: `redefinition of table 'phases.setup'` in nixpacks.toml
   - **Solution**: Merge duplicate [phases.setup] sections into a single section
   - **Cause**: Multiple [phases.setup] headers in the same configuration file
   - **Fix**: Combine aptPkgs and cmds under one [phases.setup] section

5. **ultralytics Import Error**
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
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import gc
import torch
import time
from functools import wraps

app = Flask(__name__)

# Optimized model loading for free tier
print("Loading YOLO model...")
model = YOLO("best.pt")
model.fuse()  # Fuse layers for efficiency

# Memory optimization settings
torch.set_num_threads(2)  # Balance between speed and memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Model loaded successfully!")

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Cleanup configuration
MAX_FILE_AGE = 300  # 5 minutes
MAX_FILES_IN_FOLDER = 20

def cleanup_old_files():
    """Remove old files to save disk space"""
    current_time = time.time()
    
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        if not os.path.exists(folder):
            continue
            
        files = []
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove old files
        for filepath, mtime in files:
            # Remove if older than MAX_FILE_AGE
            if current_time - mtime > MAX_FILE_AGE:
                try:
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")
        
        # If still too many files, remove oldest ones
        remaining_files = [f for f in files if os.path.exists(f[0])]
        if len(remaining_files) > MAX_FILES_IN_FOLDER:
            to_remove = remaining_files[:len(remaining_files) - MAX_FILES_IN_FOLDER]
            for filepath, _ in to_remove:
                try:
                    os.remove(filepath)
                    print(f"Removed excess file: {filepath}")
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")

# Run cleanup on startup
cleanup_old_files()

def auto_cleanup(f):
    """Decorator to automatically cleanup after request"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        finally:
            # Run cleanup periodically (10% chance per request)
            import random
            if random.random() < 0.1:
                cleanup_old_files()
    return wrapper

# Serve result images via URL
@app.route('/results/<filename>')
def serve_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "YOLO Detection API",
        "version": "2.0",
        "endpoints": {
            "/test": "GET - Test API",
            "/detect": "POST - Detect multiple images",
            "/detect_single": "POST - Detect single image",
            "/health": "GET - Health check",
            "/cleanup": "POST - Manual cleanup"
        }
    }, 200

@app.route("/test", methods=["GET"])
def test():
    return {"message": "API is working!", "status": "ok"}, 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "status": "healthy",
            "memory_used_mb": round(mem.used / 1024 / 1024, 2),
            "memory_percent": mem.percent,
            "memory_available_mb": round(mem.available / 1024 / 1024, 2)
        }, 200
    except:
        return {"status": "healthy"}, 200

@app.route("/detect", methods=["POST"])
@auto_cleanup
def detect():
    if "image" not in request.files:
        return {"error": "No images uploaded"}, 400

    files = request.files.getlist("image")
    if len(files) > 5:
        return {"error": "Maximum 5 images allowed"}, 400

    results_all = []
    combined_letters = []

    for file in files:
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        try:
            result_data = process_single_image(input_path, file.filename, request.host_url)
            
            if result_data["success"]:
                results_all.append({
                    "filename": result_data["filename"],
                    "letters": result_data["detected_letters"],
                    "detections": result_data["detections"],
                    "image_url": result_data["image_url"]
                })
                combined_letters.extend(result_data["detected_letters"])
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results_all.append({
                "filename": file.filename,
                "letters": [],
                "detections": [],
                "image_url": "",
                "error": str(e)
            })
        
        finally:
            # Immediate cleanup
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except:
                    pass
            gc.collect()

    return jsonify({
        "results": results_all,
        "combined_letters": combined_letters
    })

@app.route("/detect_single", methods=["POST"])
@auto_cleanup
def detect_single():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files["image"]
    
    if not file or file.filename == '':
        return {"error": "Empty file"}, 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        result_data = process_single_image(input_path, file.filename, request.host_url)
        return jsonify(result_data)
    
    except Exception as e:
        print(f"Error in detect_single: {e}")
        return {"error": str(e), "success": False}, 500
    
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
        gc.collect()

def process_single_image(input_path, filename, host_url):
    """
    Optimized image processing with better detection
    """
    try:
        # Load image
        img_original = cv2.imread(input_path)
        if img_original is None:
            raise ValueError("Failed to load image")
        
        img_height, img_width = img_original.shape[:2]
        
        # Adaptive image size for better detection
        max_dim = max(img_height, img_width)
        if max_dim < 400:
            imgsz = 416
        elif max_dim < 800:
            imgsz = 640
        else:
            imgsz = 640
        
        # OPTIMIZED YOLO detection
        results = model.predict(
            source=input_path,
            imgsz=imgsz,      # Adaptive size
            conf=0.15,        # Lower threshold for better detection
            iou=0.4,          # Better NMS
            max_det=50,       # Limit max detections
            verbose=False,
            device='cpu',
            half=False,
            augment=False
        )
        result = results[0]

        # Extract detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Filter small detections
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > 50:
                    detections.append({
                        "center": (int(cx), int(cy)),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf)
                    })

        # Sort left → right
        detections_sorted = sorted(detections, key=lambda d: d["center"][0])

        # Map y-coordinate to letters with adaptive tolerance
        letter_positions = {'A': 36, 'B': 59, 'C': 79, 'D': 106, 'E': 129}
        tolerance = min(14, max(8, int(img_height * 0.05)))
        
        answers = []
        detection_info = []
        
        for d in detections_sorted:
            cx, cy = d["center"]
            letter = '?'
            min_distance = float('inf')
            
            # Find closest letter
            for key, pos in letter_positions.items():
                distance = abs(cy - pos)
                if distance <= tolerance and distance < min_distance:
                    letter = key
                    min_distance = distance
            
            answers.append(letter)
            detection_info.append({
                "letter": letter,
                "center_x": int(cx),
                "center_y": int(cy),
                "bbox": {
                    "x1": int(d["bbox"][0]),
                    "y1": int(d["bbox"][1]),
                    "x2": int(d["bbox"][2]),
                    "y2": int(d["bbox"][3])
                },
                "confidence": float(d["confidence"])
            })

        # Reverse for left → right reading
        finalanswers = answers[::-1]
        detection_info_reversed = detection_info[::-1]

        # Create visualization
        img_viz = img_original.copy()
        
        # Draw grid overlay
        overlay = img_viz.copy()
        grid_spacing = 50
        grid_color = (200, 200, 200)
        
        for x in range(0, img_width, grid_spacing):
            cv2.line(overlay, (x, 0), (x, img_height), grid_color, 1, cv2.LINE_AA)
        
        for y in range(0, img_height, grid_spacing):
            cv2.line(overlay, (0, y), (img_width, y), grid_color, 1, cv2.LINE_AA)

        # Main axes
        cv2.line(overlay, (0, 0), (0, img_height), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, 0), (img_width, 0), (150, 150, 150), 1, cv2.LINE_AA)

        # Blend
        img_viz = cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0)

        # Axis labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        text_color = (80, 80, 80)
        
        for x in range(0, img_width, 100):
            cv2.putText(img_viz, str(x), (x + 2, 12), font, font_scale, 
                       text_color, 1, cv2.LINE_AA)
        
        for y in range(0, img_height, 100):
            cv2.putText(img_viz, str(y), (5, y + 12), font, font_scale, 
                       text_color, 1, cv2.LINE_AA)

        # Draw detections
        for i, d in enumerate(detections_sorted):
            cx, cy = d["center"]
            x1, y1, x2, y2 = d["bbox"]
            letter = answers[i]
            conf = d["confidence"]
            
            # Color by confidence - bright, easy to see colors
            if conf > 0.7:
                box_color = (0, 255, 0)      # Bright green
            elif conf > 0.4:
                box_color = (255, 255, 0)    # Bright yellow (cyan in BGR)
            else:
                box_color = (255, 0, 255)    # Bright magenta
            
            # Thinner bounding box (1 pixel)
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), box_color, 1)
            
            # Smaller center point
            cv2.circle(img_viz, (cx, cy), 3, (255, 0, 0), -1)
            
            # Coordinates only with cleaner background
            coord_text = f"({cx},{cy})"
            text_size = cv2.getTextSize(coord_text, font, 0.4, 1)[0]
            
            # Semi-transparent white background for better readability
            overlay_bg = img_viz.copy()
            cv2.rectangle(overlay_bg, (cx - text_size[0]//2 - 3, cy - text_size[1] - 15), 
                         (cx + text_size[0]//2 + 3, cy - 10), (255, 255, 255), -1)
            img_viz = cv2.addWeighted(overlay_bg, 0.7, img_viz, 0.3, 0)
            
            # Coordinates text in black
            cv2.putText(img_viz, coord_text, (cx - text_size[0]//2, cy - 12), font, 0.4, 
                       (0, 0, 0), 1, cv2.LINE_AA)

        # Save with compression
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, img_viz, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Build URL
        server_ip = host_url.rstrip('/')
        image_url = f"{server_ip}/results/{result_filename}"

        return {
            "success": True,
            "filename": filename,
            "detected_letters": finalanswers,
            "total_detections": len(finalanswers),
            "detections": detection_info_reversed,
            "image_url": image_url,
            "image_dimensions": {
                "width": int(img_width),
                "height": int(img_height)
            }
        }

    finally:
        # Cleanup
        if 'results' in locals():
            del results
        if 'img_original' in locals():
            del img_original
        if 'img_viz' in locals():
            del img_viz
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route("/cleanup", methods=["POST"])
def manual_cleanup():
    """Manual cleanup endpoint"""
    try:
        cleanup_old_files()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"message": "Cleanup completed successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
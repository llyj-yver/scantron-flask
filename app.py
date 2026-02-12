from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
import gc
import torch

app = Flask(__name__)

# Load YOLO model once with optimization
model = YOLO("best.pt")
model.fuse()  # Fuse layers for efficiency
torch.set_num_threads(1)  # Limit CPU threads to save memory

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Serve result images via URL
@app.route('/results/<filename>')
def serve_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/test", methods=["GET"])
def test():
    return {"message": "API is working!"}, 200

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return {"error": "No images uploaded"}, 400

    files = request.files.getlist("image")
    if len(files) > 5:
        return {"error": "Maximum 5 images allowed"}, 400

    results_all = []
    combined_letters = []  # Collect all letters from all images

    for file in files:
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        try:
            # Run YOLO detection with minimal memory settings
            results = model.predict(
                source=input_path, 
                imgsz=320,  # Keep small for free tier
                conf=0.20,
                verbose=False,  # Reduce logging overhead
                device='cpu'  # Explicitly use CPU
            )
            result = results[0]

            # Load image for processing
            img = cv2.imread(input_path)
            img_height, img_width = img.shape[:2]

            # Create transparent overlay for grid
            overlay = img.copy()

            # Draw thin grid lines
            grid_spacing = 50
            grid_color = (200, 200, 200)  # Light gray
            
            for x in range(0, img_width, grid_spacing):
                cv2.line(overlay, (x, 0), (x, img_height), grid_color, 1, cv2.LINE_AA)
            
            for y in range(0, img_height, grid_spacing):
                cv2.line(overlay, (0, y), (img_width, y), grid_color, 1, cv2.LINE_AA)

            # Main axes
            cv2.line(overlay, (0, 0), (0, img_height), (150, 150, 150), 1, cv2.LINE_AA)
            cv2.line(overlay, (0, 0), (img_width, 0), (150, 150, 150), 1, cv2.LINE_AA)

            # Blend with transparency
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

            # Thin axis labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            font_thickness = 1
            text_color = (80, 80, 80)
            
            # X-axis values
            for x in range(0, img_width, 100):
                cv2.putText(img, str(x), (x + 2, 12), font, font_scale, 
                           text_color, font_thickness, cv2.LINE_AA)
            
            # Y-axis values
            for y in range(0, img_height, 100):
                cv2.putText(img, str(y), (5, y + 12), font, font_scale, 
                           text_color, font_thickness, cv2.LINE_AA)

            # Extract bounding boxes and centers
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append({
                        "center": (cx, cy),
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(conf)
                    })

            # Sort left → right
            detections_sorted = sorted(detections, key=lambda d: d["center"][0])

            # Map y-coordinate to letters
            letter_positions = {'A':58, 'B':90, 'C':117, 'D':148, 'E':170}
            tolerance = 15
            answers = []
            detection_info = []
            
            for d in detections_sorted:
                cx, cy = d["center"]
                letter = '?'
                for key, pos in letter_positions.items():
                    if abs(cy - pos) <= tolerance:
                        letter = key
                        break
                answers.append(letter)
                detection_info.append({
                    "letter": letter,
                    "coordinates": (cx, cy),
                    "confidence": d["confidence"]
                })

            # Reverse order to match left → right
            finalanswers = answers[::-1]
            detection_info_reversed = detection_info[::-1]

            # Add to combined letters
            combined_letters.extend(finalanswers)

            # Draw detections with minimal style
            for i, d in enumerate(detections_sorted):
                cx, cy = d["center"]
                x1, y1, x2, y2 = d["bbox"]
                letter = answers[i]
                
                # Thin bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 1)
                
                # Small center point
                cv2.circle(img, (cx, cy), 3, (255, 50, 50), -1)
                
                # Letter - thin and clean
                cv2.putText(img, letter, (x1 - 15, y1 + 10), font, 0.5, 
                           (255, 50, 50), 1, cv2.LINE_AA)
                
                # Coordinates - minimal
                coord_text = f"{cx},{cy}"
                cv2.putText(img, coord_text, (cx - 20, cy - 8), font, 0.35, 
                           (50, 150, 255), 1, cv2.LINE_AA)

            # Save processed image
            result_filename = f"result_{file.filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            cv2.imwrite(result_path, img)

            # Build image URL
            server_ip = request.host_url.rstrip('/')
            image_url = f"{server_ip}/results/{result_filename}"

            # Append individual image result with coordinates
            results_all.append({
                "filename": file.filename,
                "letters": finalanswers,
                "detections": detection_info_reversed,
                "image_url": image_url
            })

        finally:
            # Critical: Clean up memory after each image
            del results
            del img
            if os.path.exists(input_path):
                os.remove(input_path)  # Delete uploaded file immediately
            gc.collect()  # Force garbage collection

    # Return JSON with individual results, coordinates, and combined letters
    return jsonify({
        "results": results_all,
        "combined_letters": combined_letters
    })
    
@app.route("/detect_single", methods=["POST"])
def detect_single():
    if "image" not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files["image"]
    
    if not file:
        return {"error": "Empty file"}, 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        # Run YOLO detection
        results = model.predict(
            source=input_path, 
            imgsz=320,
            conf=0.20,
            verbose=False,
            device='cpu'
        )
        result = results[0]

        # Load image for processing
        img = cv2.imread(input_path)
        img_height, img_width = img.shape[:2]

        # Create transparent overlay for grid
        overlay = img.copy()

        # Draw thin grid lines
        grid_spacing = 50
        grid_color = (200, 200, 200)
        
        for x in range(0, img_width, grid_spacing):
            cv2.line(overlay, (x, 0), (x, img_height), grid_color, 1, cv2.LINE_AA)
        
        for y in range(0, img_height, grid_spacing):
            cv2.line(overlay, (0, y), (img_width, y), grid_color, 1, cv2.LINE_AA)

        # Main axes
        cv2.line(overlay, (0, 0), (0, img_height), (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, 0), (img_width, 0), (150, 150, 150), 1, cv2.LINE_AA)

        # Blend with transparency
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        # Thin axis labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        font_thickness = 1
        text_color = (80, 80, 80)
        
        # X-axis values
        for x in range(0, img_width, 100):
            cv2.putText(img, str(x), (x + 2, 12), font, font_scale, 
                       text_color, font_thickness, cv2.LINE_AA)
        
        # Y-axis values
        for y in range(0, img_height, 100):
            cv2.putText(img, str(y), (5, y + 12), font, font_scale, 
                       text_color, font_thickness, cv2.LINE_AA)

        # Extract bounding boxes and centers
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    "center": (cx, cy),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf)
                })

        # Sort left → right
        detections_sorted = sorted(detections, key=lambda d: d["center"][0])

        # Map y-coordinate to letters
        letter_positions = {'A':58, 'B':90, 'C':117, 'D':148, 'E':170}
        tolerance = 15
        answers = []
        detection_info = []
        
        for d in detections_sorted:
            cx, cy = d["center"]
            letter = '?'
            for key, pos in letter_positions.items():
                if abs(cy - pos) <= tolerance:
                    letter = key
                    break
            answers.append(letter)
            detection_info.append({
                "letter": letter,
                "center_x": cx,
                "center_y": cy,
                "bbox": {
                    "x1": d["bbox"][0],
                    "y1": d["bbox"][1],
                    "x2": d["bbox"][2],
                    "y2": d["bbox"][3]
                },
                "confidence": d["confidence"]
            })

        # Reverse order to match left → right
        finalanswers = answers[::-1]
        detection_info_reversed = detection_info[::-1]

        # Draw detections
        for i, d in enumerate(detections_sorted):
            cx, cy = d["center"]
            x1, y1, x2, y2 = d["bbox"]
            letter = answers[i]
            
            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
            
            # Center point
            cv2.circle(img, (cx, cy), 4, (255, 50, 50), -1)
            
            # Letter label
            cv2.putText(img, letter, (x1 - 15, y1 + 10), font, 0.6, 
                       (255, 50, 50), 2, cv2.LINE_AA)
            
            # Coordinates
            coord_text = f"{cx},{cy}"
            cv2.putText(img, coord_text, (cx - 25, cy - 10), font, 0.4, 
                       (50, 150, 255), 1, cv2.LINE_AA)

        # Save processed image
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, img)

        # Build image URL
        server_ip = request.host_url.rstrip('/')
        image_url = f"{server_ip}/results/{result_filename}"

        # Response with detected letters and bounding boxes
        response_data = {
            "success": True,
            "filename": file.filename,
            "detected_letters": finalanswers,
            "total_detections": len(finalanswers),
            "detections": detection_info_reversed,
            "image_url": image_url,
            "image_dimensions": {
                "width": img_width,
                "height": img_height
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return {"error": str(e)}, 500

    finally:
        # Cleanup
        if 'results' in locals():
            del results
        if 'img' in locals():
            del img
        if os.path.exists(input_path):
            os.remove(input_path)
        gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
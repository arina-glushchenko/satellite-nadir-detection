import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import data_reader
import canvas
import yolo_detector

def main(file_path: str) -> None:
    """Processes a single .c file and generates a cube net visualization with bounding boxes and center points."""
    try:
        photos, face_numbers = data_reader.read_photos_from_c_file(file_path)
        print(f"[{file_path}] Read {len(photos)} faces: {face_numbers}")
        
        # Auto-select central face
        idx, threshold, counts = canvas.auto_select_center_face_index(photos)
        center_face = face_numbers[idx]
        print(f"  Auto-selected central face: {center_face} (threshold={threshold:.4f}, counts={counts})")
        
        if center_face not in range(1, 7):
            raise ValueError("Central face must be between 1 and 6")
        
        # YOLO weights path (adjust as needed)
        yolo_weights = '/media/arina/3054491A0696DA9F/work/on_git/cube_heat_detector/python_pipeline/yolov8n_1.pt'
        rgb_canvas, center_top_left, (center_w, center_h) = canvas.create_combined_canvas(
            photos, face_numbers, center_face, yolo_weights)
        
        # Run YOLO detection
        detections = yolo_detector.run_yolo_on_image(rgb_canvas, yolo_weights)
        detections_relative = []
        for d in detections:
            x1, y1, x2, y2 = d['xyxy']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            rel_cx = cx - center_top_left[0]
            rel_cy = cy - center_top_left[1]
            detections_relative.append({
                'class': d['class'],
                'conf': d['conf'],
                'center_rel': [float(rel_cx), float(rel_cy)],
                'xyxy': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Calculate vector
        x_det = center_top_left[0] + center_w / 2.0 if not detections_relative else \
                detections_relative[0]['center_rel'][0] + center_top_left[0]
        y_det = center_top_left[1] + center_h / 2.0 if not detections_relative else \
                detections_relative[0]['center_rel'][1] + center_top_left[1]
        
        x_cor = 16 - x_det
        y_cor = 12 - y_det
        r = np.sqrt(x_cor * x_cor + y_cor * y_cor)
        zenith = r / 16 * 55 * np.pi / 180
        azimuth = np.arctan2(y_cor, x_cor)
        vector = (np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith))
        
        # Visualize with bounding boxes and center points
        fig, ax = plt.subplots(figsize=(12.8, 9.6))
        ax.imshow(rgb_canvas)
        ax.axis('off')
        
        # Draw bounding boxes and center points for detections
        if detections_relative:
            for d in detections_relative:
                x1, y1, x2, y2 = d['xyxy']
                cx = d['center_rel'][0] + center_top_left[0]
                cy = d['center_rel'][1] + center_top_left[1]
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='white', facecolor='none'
                )
                ax.add_patch(rect)
                # Draw center point
                ax.scatter([cx], [cy], s=60, c='red', marker='o')
                print(f"  Detection class {d['class']} conf {d['conf']:.3f} center_rel=({d['center_rel'][0]:.2f},{d['center_rel'][1]:.2f})")
        else:
            # Draw center point of the central face
            cx = center_top_left[0] + center_w / 2.0
            cy = center_top_left[1] + center_h / 2.0
            ax.scatter([cx], [cy], s=40, c='red', marker='o')
            print(f"  No detections. center_rel=({cx - center_top_left[0]:.2f},{cy - center_top_left[1]:.2f})")
        
        print(f"  Vector: {vector}")
        
        # Adjust layout and display
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save image with annotations
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        safe_name = os.path.basename(file_path).replace(".c", ".png")
        save_path = os.path.join(output_dir, safe_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"  Saved image with annotations: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
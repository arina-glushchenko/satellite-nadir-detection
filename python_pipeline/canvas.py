import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import utils
import rotations

def add_to_canvas(sum_canvas: np.ndarray, count_canvas: np.ndarray, img: np.ndarray, y0: int, x0: int) -> None:
    """Adds an image to the canvas at specified coordinates."""
    if img is None:
        return
    H, W = sum_canvas.shape
    h, w = img.shape
    y_overlap_start = max(y0, 0)
    y_overlap_end = min(y0 + h, H)
    x_overlap_start = max(x0, 0)
    x_overlap_end = min(x0 + w, W)
    if y_overlap_start < y_overlap_end and x_overlap_start < x_overlap_end:
        img_y_start = y_overlap_start - y0
        img_y_end = img_y_start + (y_overlap_end - y_overlap_start)
        img_x_start = x_overlap_start - x0
        img_x_end = img_x_start + (x_overlap_end - x_overlap_start)
        sum_canvas[y_overlap_start:y_overlap_end, x_overlap_start:x_overlap_end] += \
            img[img_y_start:img_y_end, img_x_start:img_x_end]
        count_canvas[y_overlap_start:y_overlap_end, x_overlap_start:x_overlap_end] += 1

def auto_select_center_face_index(photos: list, percentile: int = 80) -> tuple:
    """Selects the central face based on hot pixel count."""
    all_pixels = np.concatenate([p.ravel() for p in photos])
    threshold = float(np.percentile(all_pixels, percentile))
    counts = [int(np.sum(p > threshold)) for p in photos]

    if all(c == 0 for c in counts):
        idx = int(np.argmax([np.max(p) for p in photos]))
        return idx, threshold, counts

    max_count = max(counts)
    candidates = [i for i, c in enumerate(counts) if c == max_count]
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        means = [np.mean(photos[i]) for i in candidates]
        chosen = candidates[int(np.argmax(means))]

    print(chosen)
    return chosen, threshold, counts

def create_combined_canvas(photos: list, face_numbers: list, center_face: int, yolo_weights: str = None) -> tuple:
    """Creates a combined cube net image from thermal photos."""
    face_dict = dict(zip(face_numbers, photos))
    if center_face not in face_dict:
        raise ValueError(f"Central face {center_face} not found")
    
    pos = utils.get_pos_by_center(center_face)
    rotation_func = rotations.get_rotation_strategy(center_face)
    rotated_dict = {num: rotation_func(photo, num) for num, photo in face_dict.items()}
    
    default_size = (24, 32)
    center_img = rotated_dict.get(center_face, np.zeros(default_size))
    left_img = rotated_dict.get(pos['left'], np.zeros(default_size))
    right_img = rotated_dict.get(pos['right'], np.zeros(default_size))
    up_img = rotated_dict.get(pos['up'], np.zeros(default_size))
    down_img = rotated_dict.get(pos['down'], np.zeros(default_size))
    back_img = rotated_dict.get(pos['back'], np.zeros(default_size))
    
    h, w = back_img.shape
    half_w, half_h = w // 2, h // 2
    left_half_back = back_img[:, :half_w]
    right_half_back = back_img[:, half_w:]
    back_photo_2 = np.fliplr(back_img)
    top_half_back = np.flipud(back_photo_2[:half_h, :])
    bottom_half_back = np.flipud(back_photo_2[half_h:, :])
    
    shift_x, shift_y = (3, 0) if len(face_numbers) >= 6 else (0, 0)
    
    H, W = 200, 200
    sum_canvas = np.zeros((H, W))
    count_canvas = np.zeros((H, W))
    center_h, center_w = center_img.shape
    y_base = (H - center_h) // 2
    x_center_start = (W - center_w) // 2
    
    # Left side
    left_h, left_w = left_img.shape
    y_left_start = y_base + (center_h - left_h) // 2
    x_left_start = x_center_start - left_w + shift_x
    lhb_h, lhb_w = left_half_back.shape
    y_lhb_start = y_base + (center_h - lhb_h) // 2
    x_lhb_start = x_left_start - lhb_w + shift_x
    
    # Right side
    right_h, right_w = right_img.shape
    y_right_start = y_base + (center_h - right_h) // 2
    x_right_start = x_center_start + center_w - shift_x
    rhb_h, rhb_w = right_half_back.shape
    y_rhb_start = y_base + (center_h - rhb_h) // 2
    x_rhb_start = x_right_start + right_w - shift_x
    
    # Top side
    up_h, up_w = up_img.shape
    x_up_start = x_center_start + (center_w - up_w) // 2
    y_up_start = y_base - up_h + shift_y
    thb_h, thb_w = top_half_back.shape
    x_thb_start = x_center_start + (center_w - thb_w) // 2
    y_thb_start = y_up_start - thb_h + shift_y
    
    # Bottom side
    down_h, down_w = down_img.shape
    x_down_start = x_center_start + (center_w - down_w) // 2
    y_down_start = y_base + center_h - shift_y
    bhb_h, bhb_w = bottom_half_back.shape
    x_bhb_start = x_center_start + (center_w - bhb_w) // 2
    y_bhb_start = y_down_start + down_h - shift_y
    
    if len(face_numbers) == 6 and shift_x > 0:
        # Horizontal averaging
        l_edge = left_half_back[:, -shift_x:]
        li_edge = left_img[:, :shift_x]
        avg = (l_edge + li_edge) / 2
        left_half_back[:, -shift_x:] = avg
        left_img[:, :shift_x] = avg
        
        li_edge2 = left_img[:, -shift_x:]
        ci_edge = center_img[:, :shift_x]
        avg2 = (li_edge2 + ci_edge) / 2
        left_img[:, -shift_x:] = avg2
        center_img[:, :shift_x] = avg2
        
        ci_edge = center_img[:, -shift_x:]
        ri_edge = right_img[:, :shift_x]
        avg3 = (ci_edge + ri_edge) / 2
        center_img[:, -shift_x:] = avg3
        right_img[:, :shift_x] = avg3
        
        ri_edge2 = right_img[:, -shift_x:]
        rb_edge = right_half_back[:, :shift_x]
        avg4 = (ri_edge2 + rb_edge) / 2
        right_img[:, -shift_x:] = avg4
        right_half_back[:, :shift_x] = avg4
    
    if len(face_numbers) == 6 and shift_y > 0:
        # Vertical averaging
        tb_edge = top_half_back[-shift_y:, :]
        ui_edge = up_img[:shift_y, :]
        avg5 = (tb_edge + ui_edge) / 2
        top_half_back[-shift_y:, :] = avg5
        up_img[:shift_y, :] = avg5
        
        ui_edge2 = up_img[-shift_y:, :]
        ci_edge_top = center_img[:shift_y, :]
        avg6 = (ui_edge2 + ci_edge_top) / 2
        up_img[-shift_y:, :] = avg6
        center_img[:shift_y, :] = avg6
        
        ci_edge_bot = center_img[-shift_y:, :]
        di_edge = down_img[:shift_y, :]
        avg7 = (ci_edge_bot + di_edge) / 2
        center_img[-shift_y:, :] = avg7
        down_img[:shift_y, :] = avg7
        
        di_edge2 = down_img[-shift_y:, :]
        bb_edge = bottom_half_back[:shift_y, :]
        avg8 = (di_edge2 + bb_edge) / 2
        down_img[-shift_y:, :] = avg8
        bottom_half_back[:shift_y, :] = avg8
    
    # Add faces to canvas
    add_to_canvas(sum_canvas, count_canvas, center_img, y_base, x_center_start)
    add_to_canvas(sum_canvas, count_canvas, left_half_back, y_lhb_start, x_lhb_start)
    add_to_canvas(sum_canvas, count_canvas, left_img, y_left_start, x_left_start)
    add_to_canvas(sum_canvas, count_canvas, right_img, y_right_start, x_right_start)
    add_to_canvas(sum_canvas, count_canvas, right_half_back, y_rhb_start, x_rhb_start)
    add_to_canvas(sum_canvas, count_canvas, top_half_back, y_thb_start, x_thb_start)
    add_to_canvas(sum_canvas, count_canvas, up_img, y_up_start, x_up_start)
    add_to_canvas(sum_canvas, count_canvas, down_img, y_down_start, x_down_start)
    add_to_canvas(sum_canvas, count_canvas, bottom_half_back, y_bhb_start, x_bhb_start)
    
    value_canvas = np.where(count_canvas > 0, sum_canvas / count_canvas, 0)
    vmin = np.min(value_canvas[count_canvas > 0]) if np.any(count_canvas > 0) else 0
    vmax = np.max(value_canvas[count_canvas > 0]) if np.any(count_canvas > 0) else 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['inferno']
    rgb_canvas = np.zeros((H, W, 3), dtype=np.float32)
    valid_mask = count_canvas > 0
    rgb_canvas[valid_mask] = cmap(norm(value_canvas[valid_mask]))[..., :3]
    
    y_used = np.where(count_canvas.sum(axis=1) > 0)[0]
    x_used = np.where(count_canvas.sum(axis=0) > 0)[0]
    y0_crop = x0_crop = 0
    if y_used.size > 0 and x_used.size > 0:
        y0_crop = y_used[0]
        x0_crop = x_used[0]
        rgb_canvas = rgb_canvas[y_used[0]:y_used[-1] + 1, x_used[0]:x_used[-1] + 1]
    
    rgb_canvas_uint8 = (np.clip(rgb_canvas, 0, 1) * 255).astype(np.uint8)
    center_top_left = (x_center_start - x0_crop, y_base - y0_crop)
    
    return rgb_canvas_uint8, center_top_left, (center_w, center_h)
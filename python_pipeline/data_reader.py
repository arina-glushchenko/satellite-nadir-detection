import re
import numpy as np

def read_photos_from_c_file(file_path: str) -> tuple:
    """Reads photos and face numbers from a .c file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove 'f' or 'F' from floating-point numbers
    content = re.sub(r'([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)[fF]', r'\1', content)
    
    # Extract face numbers from comments
    face_comments = re.findall(r'//\s*hs\s*(\d+)\s*photo', content, re.IGNORECASE)
    face_numbers = [int(num) for num in face_comments]

    # Extract arrays
    arrays = re.findall(r'{([^{}]*)}', content)
    if not arrays:
        raise ValueError("No arrays found in the file.")

    photos, valid_face_numbers = [], []
    for i, array_str in enumerate(arrays):
        numbers = [float(num.strip()) for num in array_str.split(',') if num.strip()]
        if len(numbers) == 768:
            photos.append(np.array(numbers).reshape((24, 32)))
            face_num = face_numbers[i] if i < len(face_numbers) else i + 1
            valid_face_numbers.append(face_num)

    return photos, valid_face_numbers
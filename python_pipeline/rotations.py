import numpy as np

def rotate_face_1(photo: np.ndarray, face_number: int) -> np.ndarray:
    return np.rot90(photo, 2) if face_number == 3 else photo

def rotate_face_2(photo: np.ndarray, face_number: int) -> np.ndarray:
    return np.rot90(photo, 1) if face_number in (1, 2, 3, 4) else photo

def rotate_face_3(photo: np.ndarray, face_number: int) -> np.ndarray:
    if face_number in (2, 3):
        return np.rot90(photo, 2)
    if face_number == 4:
        return np.fliplr(photo)
    return photo

def rotate_face_4(photo: np.ndarray, face_number: int) -> np.ndarray:
    if face_number in (1, 2, 3, 4):
        return np.rot90(photo, 3)
    if face_number == 6:
        return np.rot90(photo, 2)
    return photo

def rotate_face_5(photo: np.ndarray, face_number: int) -> np.ndarray:
    if face_number == 2:
        return np.rot90(photo, 1)
    if face_number == 3:
        return np.rot90(photo, 2)
    if face_number == 4:
        return np.rot90(photo, 3)
    if face_number == 6:
        return np.rot90(photo, 2)
    return photo

def rotate_face_6(photo: np.ndarray, face_number: int) -> np.ndarray:
    if face_number == 2:
        return np.rot90(photo, 3)
    if face_number == 3:
        return np.rot90(photo, 2)
    if face_number == 4:
        return np.rot90(photo, 1)
    if face_number == 6:
        return np.rot90(photo, 2)
    return photo

def get_rotation_strategy(center_face: int) -> callable:
    """Returns the rotation function for the specified central face."""
    strategies = {
        1: rotate_face_1,
        2: rotate_face_2,
        3: rotate_face_3,
        4: rotate_face_4,
        5: rotate_face_5,
        6: rotate_face_6
    }
    return strategies[center_face]
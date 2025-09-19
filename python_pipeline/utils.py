import numpy as np

def get_pos_by_center(center_face: int) -> dict:
    """Returns face positions relative to the central face for cube net assembly."""
    face_adjacency = {
        1: [1, 4, 2, 6, 5, 3],
        2: [2, 5, 6, 1, 3, 4],
        3: [3, 4, 2, 5, 6, 1],
        4: [4, 6, 5, 1, 3, 2],
        5: [5, 4, 2, 1, 3, 6],
        6: [6, 4, 2, 3, 1, 5]
    }
    if center_face not in face_adjacency:
        raise ValueError("Central face must be between 1 and 6")
    layout = face_adjacency[center_face]
    return {
        'center': layout[0],
        'up': layout[1],
        'down': layout[2],
        'left': layout[3],
        'right': layout[4],
        'back': layout[5]
    }
"""Project control-frame axes onto a private preview image."""
import numpy as np

def project(points, camera, width, height):
    points = np.asarray(points, dtype=float)
    position = np.asarray(camera['position'], dtype=float).reshape(3)
    rotation = np.asarray(camera['rotation'], dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(np.r_[points.reshape(-1), position, rotation.reshape(-1)])):
        raise ValueError('Non-finite preview geometry')
    local = (points - position) @ rotation
    forward = camera.get('forward_axis', -1)
    if forward not in (-1, 1):
        raise ValueError('Camera forward_axis must be -1 or 1')
    depth = forward * local[:, 2]
    valid = depth > .001
    pixels = np.full((len(points), 2), np.nan)
    if 'intrinsics' in camera:
        intrinsics = np.asarray(camera['intrinsics'], dtype=float).reshape(3, 3)
        source_size = np.asarray(camera['image_size'], dtype=float).reshape(2)
        if (not np.all(np.isfinite(intrinsics)) or not np.all(np.isfinite(source_size))
                or np.any(source_size <= 0) or abs(np.linalg.det(intrinsics)) < 1e-12):
            raise ValueError('Invalid preview camera intrinsics')
        homogeneous = local[valid] @ intrinsics.T
        pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
        pixels[valid] = (pixels[valid] + .5) * np.array([width, height]) / source_size - .5
    else:
        fovy = float(camera['fovy'])
        if not 0 < fovy < 180:
            raise ValueError('Invalid preview camera field of view')
        focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
        pixels[valid, 0] = (width - 1) / 2 + focal * local[valid, 0] / depth[valid]
        pixels[valid, 1] = (height - 1) / 2 - focal * local[valid, 1] / depth[valid]
    if camera.get('flip_x'):
        pixels[:, 0] = width - 1 - pixels[:, 0]
    if camera.get('flip_y'):
        pixels[:, 1] = height - 1 - pixels[:, 1]
    return pixels


def axes_markers(camera, effectors, image_size, length=.08):
    """Project 3D axes into portable, normalized 2D markers; no drawing here."""
    width, height = image_size
    result = []
    for effector in effectors:
        try:
            origin = np.asarray(effector['position'], dtype=float).reshape(3)
            rotation = np.asarray(effector.get('rotation', np.eye(3)), dtype=float).reshape(3, 3)
            points = np.vstack([origin, origin + rotation.T * length])
            pixels = project(points, camera, width, height)
            if not np.all(np.isfinite(pixels[0])):
                continue
            normalized = pixels / [max(1, width-1), max(1, height-1)]
            coordinates = [point.tolist() if np.all(np.isfinite(point)) else None for point in normalized]
            camera_z = np.asarray(camera['rotation']).reshape(3, 3)[:, 2]
            toward = (-camera.get('forward_axis', -1) * (rotation.T @ camera_z) > 0).tolist()
            result.append(dict(type='axes', points=coordinates, toward_viewer=toward,
                               label=effector.get('label', '')))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return result

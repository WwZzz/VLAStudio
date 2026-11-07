"""Project control-frame axes onto a private preview image."""
import cv2
import numpy as np

AXES = (("X", (48, 48, 225)), ("Y", (55, 180, 35)), ("Z", (235, 105, 40)))


def project(points, camera, width, height):
    points = np.asarray(points, dtype=float)
    local = (points - np.asarray(camera['position'])) @ np.asarray(camera['rotation']).reshape(3, 3)
    depth = -local[:, 2]
    focal = height / (2 * np.tan(np.deg2rad(camera['fovy']) / 2))
    valid = depth > .001
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid, 0] = (width - 1) / 2 + focal * local[valid, 0] / depth[valid]
    pixels[valid, 1] = (height - 1) / 2 - focal * local[valid, 1] / depth[valid]
    if camera.get('flip_x'):
        pixels[:, 0] = width - 1 - pixels[:, 0]
    if camera.get('flip_y'):
        pixels[:, 1] = height - 1 - pixels[:, 1]
    return pixels


def draw_axes(image, camera, effectors):
    """Draw only on the preview's owned, resized BGR buffer."""
    height, width = image.shape[:2]
    for effector in effectors:
        origin = np.asarray(effector['position'], dtype=float)
        points = np.vstack([origin, origin + np.eye(3) * .08])
        pixels = project(points, camera, width, height)
        if not np.all(np.isfinite(pixels[0])):
            continue
        start = tuple(np.rint(pixels[0]).astype(int))
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            continue
        cv2.circle(image, start, 3, (255, 255, 255), -1, cv2.LINE_AA)
        for index, (label, color) in enumerate(AXES, 1):
            if not np.all(np.isfinite(pixels[index])):
                continue
            end = tuple(np.rint(pixels[index]).astype(int))
            if np.linalg.norm(pixels[index] - pixels[0]) < 5:
                # An axis along the viewing direction has almost no 2D arrow length.
                cv2.circle(image, start, 7, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(image, start, 7, color, 2, cv2.LINE_AA)
                camera_z = np.asarray(camera['rotation']).reshape(3, 3)[:, 2]
                if np.dot(points[index] - points[0], camera_z) > 0:
                    cv2.circle(image, start, 2, color, -1, cv2.LINE_AA)
                else:
                    for sign in (-1, 1):
                        cv2.line(image, (start[0]-3, start[1]-3*sign), (start[0]+3, start[1]+3*sign), color, 1, cv2.LINE_AA)
                end = (start[0] + 9, start[1] + 17)
            else:
                cv2.arrowedLine(image, start, end, (255, 255, 255), 4, cv2.LINE_AA, tipLength=.2)
                cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=.2)
            text = '+' + label
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0]
            text_position = (int(np.clip(end[0] + 3, 2, width - size[0] - 2)),
                             int(np.clip(end[1] - 3, size[1] + 2, height - 3)))
            cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        cv2.putText(image, effector.get('label', ''), (start[0] - 14, start[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)

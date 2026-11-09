"""Typed 2D preview drawing. No simulator imports, handles or kinematics."""
import cv2
import numpy as np


def _point(value, image):
    point = np.asarray(value, dtype=float).reshape(2)
    if not np.all(np.isfinite(point)):
        raise ValueError('Non-finite marker point')
    return tuple(np.rint(np.clip(point, -1e4, 1e4) * [image.shape[1]-1, image.shape[0]-1]).astype(int))


def _inside(point, image):
    return 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]


def _text(image, text, point):
    text = str(text)[:64]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0]
    x = int(np.clip(point[0], 2, max(2, image.shape[1]-size[0]-2)))
    y = int(np.clip(point[1], size[1]+2, image.shape[0]-3))
    for color, thickness in [((255,255,255), 3), ((35,35,35), 1)]:
        cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .4, color, thickness, cv2.LINE_AA)


def draw_axes_marker(image, marker):
    points = marker['points']
    if len(points) != 4:
        return
    start = _point(points[0], image)
    if not _inside(start, image):
        return
    endpoints = [None if point is None else _point(point, image) for point in points[1:]]
    toward = marker.get('toward_viewer', [False]*3)
    if len(toward) != 3:
        return
    cv2.circle(image, start, 3, (255,255,255), -1, cv2.LINE_AA)
    for i, (label, color) in enumerate([('X',(48,48,225)), ('Y',(55,180,35)), ('Z',(235,105,40))]):
        end = endpoints[i]
        if end is None:
            continue
        if np.linalg.norm(np.asarray(end)-start) < 5:
            cv2.circle(image, start, 7, (255,255,255), 3, cv2.LINE_AA)
            cv2.circle(image, start, 7, color, 2, cv2.LINE_AA)
            if toward[i]:
                cv2.circle(image, start, 2, color, -1, cv2.LINE_AA)
            else:
                for sign in (-1,1):
                    cv2.line(image, (start[0]-3,start[1]-3*sign), (start[0]+3,start[1]+3*sign),color,1,cv2.LINE_AA)
            end = (start[0]+9,start[1]+17)
        else:
            cv2.arrowedLine(image,start,end,(255,255,255),4,cv2.LINE_AA,tipLength=.2)
            cv2.arrowedLine(image,start,end,color,2,cv2.LINE_AA,tipLength=.2)
        size = cv2.getTextSize('+'+label,cv2.FONT_HERSHEY_SIMPLEX,.4,1)[0]
        position=(int(np.clip(end[0]+3,2,max(2,image.shape[1]-size[0]-2))),
                  int(np.clip(end[1]-3,size[1]+2,image.shape[0]-3)))
        cv2.putText(image,'+'+label,position,cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(image,'+'+label,position,cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    _text(image,marker.get('label',''),(start[0]-14,start[1]-9))


def draw_joint_marker(image, marker):
    center = _point(marker['center'], image)
    if not _inside(center,image):
        return
    radius = float(marker.get('radius',.025))
    direction = np.asarray(marker['positive_direction'],dtype=float).reshape(2).copy()
    if not 0 < radius <= .25 or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-8:
        return
    radius = max(4,int(radius*min(image.shape[:2])))
    direction /= np.linalg.norm(direction)
    positive=tuple(np.rint(np.asarray(center)+direction*(radius+12)).astype(int))
    negative=tuple(np.rint(np.asarray(center)-direction*(radius+12)).astype(int))
    cv2.circle(image,center,radius,(255,255,255),4,cv2.LINE_AA)
    cv2.circle(image,center,radius,(40,170,245),2,cv2.LINE_AA)
    cv2.arrowedLine(image,negative,positive,(255,255,255),4,cv2.LINE_AA,tipLength=.2)
    cv2.arrowedLine(image,negative,positive,(40,170,245),2,cv2.LINE_AA,tipLength=.2)
    _text(image,'+',positive);_text(image,'-',negative)
    label=marker.get('label',marker.get('id',''))
    if 'value' in marker:
        label=f"{label}: {float(marker['value']):.3f}"
    _text(image,label,(center[0]+radius+6,center[1]+radius+20))


RENDERERS = {'axes':draw_axes_marker,'joint':draw_joint_marker}


def register_marker_renderer(kind, renderer):
    if not isinstance(kind,str) or not callable(renderer):
        raise ValueError('Marker renderer needs a type name and callable')
    RENDERERS[kind]=renderer


def draw_markers(image, markers):
    """Return an annotated BGR copy; unknown or invalid markers are omitted."""
    result=image.copy()
    if not isinstance(markers,(list,tuple)):
        return result
    for marker in markers:
        if not isinstance(marker,dict):
            continue
        kind=marker.get('type')
        renderer=RENDERERS.get(kind) if isinstance(kind,str) else None
        if renderer is None:
            continue
        try:
            renderer(result,marker)
        except Exception:
            continue
    return result

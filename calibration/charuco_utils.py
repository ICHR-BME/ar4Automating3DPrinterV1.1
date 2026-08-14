"""OpenCV-version-independent ChArUco helpers used by calibration tools."""

from __future__ import annotations

import cv2


def detector_parameters():
    factory = getattr(cv2.aruco, "DetectorParameters_create", None)
    return factory() if factory is not None else cv2.aruco.DetectorParameters()


def create_board(squares_x, squares_y, square_length, marker_length,
                 dictionary):
    factory = getattr(cv2.aruco, "CharucoBoard_create", None)
    if factory is not None:
        return factory(squares_x, squares_y, square_length, marker_length,
                       dictionary)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary)


def detect(gray, board, dictionary, parameters=None):
    """Return marker corners/ids and interpolated ChArUco corners/ids."""
    parameters = parameters or detector_parameters()
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=parameters)
    charuco_corners = charuco_ids = None
    if marker_ids is not None and len(marker_ids) >= 2:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board)
    return (marker_corners, marker_ids, rejected,
            charuco_corners, charuco_ids)

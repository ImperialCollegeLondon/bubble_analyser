import cv2
import numpy as np

from bubble_analyser.processing.circle_handler import EllipseHandler

FILTERS = {
    "max_eccentricity": 0.85,
    "min_solidity": 0.9,
    "min_circularity": 0.5,
    "min_size": 0.5,  # mm
    "max_size": 10.0,  # mm
}


def _fit(labels: np.ndarray) -> EllipseHandler:
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    handler = EllipseHandler(labels, image, px2mm_display=10.0, resample=1.0)  # 10 px per mm
    handler.load_filter_params(FILTERS)
    handler.filter_labels_properties()
    handler.fill_ellipse_labels()
    return handler


def _disk(labels: np.ndarray, value: int, center: tuple[int, int], radius: int) -> None:
    cv2.circle(labels, center, radius, value, -1)


def test_background_label_is_read_from_the_border():
    labels = np.full((60, 60), 7, dtype=np.int32)
    labels[20:30, 20:30] = 3

    assert EllipseHandler().detect_background_label(labels) == 7


def test_round_bubble_stays_an_ellipse():
    labels = np.ones((200, 200), dtype=np.int32)
    _disk(labels, 2, (100, 100), 15)

    handler = _fit(labels)

    assert len(handler.ellipses) == 1
    assert handler.ellipse_polygons == [None]


def test_irregular_mask_is_kept_as_a_polygon():
    labels = np.ones((200, 200), dtype=np.int32)
    labels[60:140, 60:80] = 2  # an L shape: low solidity, so no ellipse fits it
    labels[120:140, 60:140] = 2

    handler = _fit(labels)

    assert len(handler.ellipses) == 1
    assert handler.ellipse_polygons[0] is not None and len(handler.ellipse_polygons[0]) >= 3


def test_noise_below_the_minimum_size_is_still_removed():
    labels = np.ones((200, 200), dtype=np.int32)
    _disk(labels, 2, (100, 100), 2)  # 4 px diameter = 0.4 mm < 0.5 mm

    handler = _fit(labels)

    assert handler.ellipses == []


def test_polygons_stay_aligned_with_the_ellipses():
    labels = np.ones((300, 300), dtype=np.int32)
    _disk(labels, 2, (60, 60), 15)
    labels[150:230, 150:170] = 3
    labels[210:230, 150:230] = 3

    handler = _fit(labels)

    assert len(handler.ellipse_polygons) == len(handler.ellipses) == 2
    assert sum(polygon is not None for polygon in handler.ellipse_polygons) == 1

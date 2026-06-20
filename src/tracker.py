"""
Lightweight multi-object tracker for stationary-vehicle detection.

We don't need a heavy tracker (DeepSORT etc.) for this use case —
a simple centroid tracker is enough to follow vehicles across frames
and measure how long they've stayed in roughly the same spot.
"""

import numpy as np
from collections import OrderedDict


class CentroidTracker:
    """
    Assigns persistent IDs to detections across frames by matching
    centroids frame-to-frame (nearest neighbour, with a max distance
    cutoff so unrelated vehicles don't get merged).
    """

    def __init__(self, max_disappeared: int = 30, max_distance: int = 60):
        self.next_object_id = 0
        self.objects = OrderedDict()        # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()         # id -> last bbox
        self.disappeared = OrderedDict()    # id -> frames since last seen
        self.first_seen_frame = OrderedDict()   # id -> frame index when first detected
        self.position_history = OrderedDict()   # id -> list of centroids (for movement check)

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.bboxes[oid] = bbox
        self.disappeared[oid] = 0
        self.first_seen_frame[oid] = self.frame_count
        self.position_history[oid] = [centroid]
        self.next_object_id += 1
        return oid

    def deregister(self, oid):
        del self.objects[oid]
        del self.bboxes[oid]
        del self.disappeared[oid]
        del self.first_seen_frame[oid]
        del self.position_history[oid]

    def update(self, detections: list[dict]) -> dict:
        """
        detections: list of {"bbox": (x1,y1,x2,y2), ...}
        Returns: dict of {object_id: detection_dict} for currently visible objects
        """
        self.frame_count += 1

        if len(detections) == 0:
            # Mark everyone as disappeared this frame
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {}

        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            result = {}
            for i, det in enumerate(detections):
                oid = self.register(input_centroids[i], det["bbox"])
                result[oid] = det
            return result

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Distance matrix: existing objects x new detections
        D = np.zeros((len(object_centroids), len(input_centroids)))
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # Greedy nearest-neighbour matching
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        result = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.bboxes[oid] = detections[col]["bbox"]
            self.disappeared[oid] = 0
            self.position_history[oid].append(input_centroids[col])
            if len(self.position_history[oid]) > 90:   # cap history length
                self.position_history[oid] = self.position_history[oid][-90:]

            result[oid] = detections[col]
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_centroids))) - used_rows
        unused_cols = set(range(len(input_centroids))) - used_cols

        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for col in unused_cols:
            oid = self.register(input_centroids[col], detections[col]["bbox"])
            result[oid] = detections[col]

        return result

    def get_stationary_duration(self, oid: int, movement_threshold: int = 15) -> int:
        """
        Returns how many consecutive recent frames this object has stayed
        within `movement_threshold` pixels of its position (i.e. not moving).
        """
        history = self.position_history.get(oid, [])
        if len(history) < 2:
            return 0

        stationary_frames = 1
        ref = history[-1]
        for pos in reversed(history[:-1]):
            dist = np.linalg.norm(np.array(pos) - np.array(ref))
            if dist <= movement_threshold:
                stationary_frames += 1
            else:
                break
        return stationary_frames

"""Bidirectional line-crossing counter for traffic and pedestrian monitoring."""
from typing import Tuple, Set

class LineCrossCounter:
    def __init__(self, line_p1: Tuple[int, int], line_p2: Tuple[int, int]):
        self.p1 = line_p1
        self.p2 = line_p2
        self.in_count = 0
        self.out_count = 0
        self._history: dict[int, list[Tuple[int, int]]] = {}
        self._counted_tracks: Set[int] = set()

    def _ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def update(self, track_id: int, centroid: Tuple[int, int]):
        if track_id not in self._history:
            self._history[track_id] = []
        self._history[track_id].append(centroid)

        if len(self._history[track_id]) >= 2 and track_id not in self._counted_tracks:
            prev_pt = self._history[track_id][-2]
            if self._intersect(prev_pt, centroid, self.p1, self.p2):
                if centroid[1] > prev_pt[1]:
                    self.in_count += 1
                else:
                    self.out_count += 1
                self._counted_tracks.add(track_id)

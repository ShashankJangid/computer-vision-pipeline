from src.heatmap import HeatmapGenerator
from src.line_counter import LineCrossCounter

def test_heatmap_initialization():
    hm = HeatmapGenerator(width=640, height=480)
    assert hm.density.shape == (480, 640)

def test_line_counter_initialization():
    counter = LineCrossCounter((0, 240), (640, 240))
    assert counter.in_count == 0
    assert counter.out_count == 0

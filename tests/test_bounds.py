from asesurfacefinder import SampleBounds

def test_bounds_hemispheroid():
    b = SampleBounds(0.5, 1.0, 2.0)
    assert b.is_ovoid is False
    assert len(b.z_bounds) == 2

def test_bounds_ovoid():
    b = SampleBounds(0.3, -0.1, 1.4, z_mid=0.6)
    assert b.is_ovoid
    assert len(b.z_bounds) == 3
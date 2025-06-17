class SampleBounds:
    def __init__(self, r_max: float, z_min: float, z_max: float, z_mid: float=None):
        if r_max < 0:
            raise ValueError('Sampling r_max must be greater than or equal to zero.')
        if z_mid is not None:
            if not (z_min < z_mid and z_mid < z_max):
                raise ValueError('Sampling Z-bounds incorrectly ordered, must be z_min < z_mid < z_max.')
        else:
            if not z_min < z_max:
                raise ValueError('Sampling Z-bounds incorrectly ordered, must be z_min < z_max.')
            
        self.r_max = r_max
        self.z_min = z_min
        self.z_mid = z_mid
        self.z_max = z_max
        self.is_ovoid = False if z_mid is None else True
        self.z_bounds = (self.z_min, self.z_mid, self.z_max) if self.is_ovoid else (self.z_min, self.z_max)

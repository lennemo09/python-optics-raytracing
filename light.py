class OpticalProperties:
    def __init__(self, color, refractive_index):
        self.color = color
        self.refractive_index = refractive_index

# 620-780 nm
red_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.488,
    'water' : 1.330
}

# 585-620 nm
orange_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.490,
    'water' : 1.332
}

# 570-585 nm
yellow_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.497,
    'water' : 1.333
}

# 490 - 570 nm
green_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.495,
    'water' : 1.335
}

# 440 - 490 nm
blue_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.502,
    'water' : 1.337
}

# 420 - 440 nm
indigo_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.504,
    'water' : 1.338
}

# 400 - 440 nm
violet_light_refractive_index = {
    'air'   : 1.0003,
    'glass' : 1.508,
    'water' : 1.339
}

red_light = OpticalProperties('red', red_light_refractive_index)
yellow_light = OpticalProperties('yellow', yellow_light_refractive_index)
orange_light = OpticalProperties('orange', orange_light_refractive_index)
green_light = OpticalProperties('green', green_light_refractive_index)

import numpy as np
import matplotlib.pyplot as plt

import pattern
from generator import ImageGenerator
from pattern import Checker, Circle

label_path = r"E:\DL\exercise0_material\exercise0_material\src_to_implement\data\Labels.json"
file_path = r"E:\DL\exercise0_material\exercise0_material\src_to_implement\data\exercise_data"
if __name__ == '__main__':
    checker = Checker(250, 25)
    checker.show()

    circle = pattern.Circle(1024, 200, (512, 256))
    circle.show()
    #
    spe = pattern.Spectrum(100)
    #
    spe.show()

    g1 = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    # generator.show()
    g2 = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)

    g2.show()
    print('')
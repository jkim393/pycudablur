#!/usr/bin/env python3
import math
import multiprocessing
import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler
import sys
import PIL.Image

DEFAULT_STANDARD_DEVIATION = 1.0
DEFAULT_FILTER_WIDTH = 5
BLOCK_SIZE = 16

def blur(source_array, standard_deviation, filter_width):
    result_array = np.empty_like(source_array)
    red_channel = source_array[:, :, 0].copy()
    green_channel = source_array[:, :, 1].copy()
    blue_channel = source_array[:, :, 2].copy()

    height, width = source_array.shape[:2]

    dim_grid_x = math.ceil(width / BLOCK_SIZE)
    dim_grid_y = math.ceil(height / BLOCK_SIZE)

    gaussian_kernel = create_gaussian_kernel(
        filter_width,
        standard_deviation
    )

    mod = compiler.SourceModule(open('./blur.cu').read())
    gaussian_blur = mod.get_function('gaussian_blur')

    for channel in (red_channel, green_channel, blue_channel):
        gaussian_blur(
            driver.In(channel),
            driver.Out(channel),
            np.uint32(width),
            np.uint32(height),
            driver.In(gaussian_kernel),
            np.uint32(filter_width),
            block=(BLOCK_SIZE, BLOCK_SIZE, 1),
            grid=(dim_grid_x, dim_grid_y)
        )

    result_array[:, :, 0] = red_channel
    result_array[:, :, 1] = green_channel
    result_array[:, :, 2] = blue_channel

    return result_array


def create_gaussian_kernel(filter_width, standard_deviation):
    matrix = np.empty((filter_width, filter_width), np.float32)
    filter_half = filter_width // 2
    for i in range(-filter_half, filter_half + 1):
        for j in range(-filter_half, filter_half + 1):
            matrix[i + filter_half][j + filter_half] = (
                np.exp(-(i**2 + j**2) / (2 * standard_deviation**2))
                / (2 * np.pi * standard_deviation**2)
            )

    return matrix / matrix.sum()



#start of the script

image_source = sys.argv[1]
image_destination = sys.argv[2]
result_array = None
image = PIL.Image.open(image_source)
source_array = np.array(image)
standard_deviation = 1.0
filter_width = 5
result_array = blur(
                     source_array,
                     standard_deviation,
                     filter_width
)
PIL.Image.fromarray(result_array).save(image_destination)

import numpy as np
import scipy as sp
import scipy.misc


def add_noise(x, local_random,
                  std=0.6,
                  uniform_max=3.464,
                  continuous_noise=0,
                  use_spatially_varying_uniform_on_top=1,
                  min_spatially_continuous_noise_factor=0.01,
                  max_spatially_continuous_noise_factor=0.5,
                  clean_img_prob=0.0,
                  clip_input=0, clip_input_bound=10.0,
            ):
    """
    Perturb an input m*n*3 image x by adding Gaussian noise with spatially-varying standard deviation
    and smoothing the image by downsampling and upsampling with nearest neighbor algorithm.  The
    smoothing method generates blurred and blocky outputs.
    """

    def actually_add_noise(x):
        y = x
        x_shape = x.shape

        if len(x_shape) == 2:
            x_shape.append(1)

        if continuous_noise > 0:
            rand_std = local_random.rand() * (std - 0.05) + 0.05
            noise = local_random.randn(x_shape[0], x_shape[1], x_shape[2]) * rand_std
        else:
            noise = local_random.randn(x_shape[0], x_shape[1], x_shape[2]) * std

        if use_spatially_varying_uniform_on_top > 0:
            for channel in range(x_shape[2]):
                low_res_row = np.amax( [np.round(float(x_shape[0]) * local_random.rand() * (max_spatially_continuous_noise_factor - min_spatially_continuous_noise_factor)).astype(int), 1])
                low_res_col = np.amax( [np.round(float(x_shape[1]) * local_random.rand() * (max_spatially_continuous_noise_factor - min_spatially_continuous_noise_factor)).astype(int), 1])


                lowres_noise_map = local_random.rand(low_res_row,low_res_col) * 2 * uniform_max - uniform_max
                highres_noise_map = sp.misc.imresize(lowres_noise_map, [x_shape[0], x_shape[1]], interp='bicubic', mode='F')
                noise[:,:,channel] *= highres_noise_map

        y += noise

        return y

    def create_blocky_image(x, min_resize_ratio):
        ratio = local_random.rand() * (0.95 - min_resize_ratio) + min_resize_ratio
        tmp = sp.misc.imresize(x, ratio, interp='nearest')
        y = sp.misc.imresize(tmp, [x.shape[0], x.shape[1]], interp='nearest').astype(float) / 255.0

        return y

    y = np.copy(x)

    # the probability to smooth the input before adding noise
    prob_block = 0.3

    r = local_random.rand()
    if r < prob_block:
        y = create_blocky_image(x, min_resize_ratio=0.2)

    result = actually_add_noise(y)

    return result


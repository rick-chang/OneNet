
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
import matplotlib
import matplotlib.pyplot as plt
from vec import vec
import timeit
import pywt
import os

def vec(x):
    return x.ravel(order='F')


def sigmoid(x):
    return 1/(1+np.exp(-x))


def wavelet_transform(x):
    w_coeffs_rgb = [] # np.zeros(x.shape[3], np.prod(x.shape))
    for i in range(x.shape[3]):
        w_coeffs_list = pywt.wavedec2(x[0,:,:,i], 'db4', level=None, mode='periodization')
        w_coeffs, coeff_slices = pywt.coeffs_to_array(w_coeffs_list)
        w_coeffs_rgb.append(w_coeffs)

    w_coeffs_rgb = np.array(w_coeffs_rgb)
    return w_coeffs_rgb, coeff_slices


def inverse_wavelet_transform(w_coeffs_rgb, coeff_slices, x_shape):
    x_hat = np.zeros(x_shape)
    for i in range(w_coeffs_rgb.shape[0]):
        w_coeffs_list = pywt.array_to_coeffs(w_coeffs_rgb[i,:,:], coeff_slices)
        x_hat[0,:,:,i] = pywt.waverecn(w_coeffs_list, wavelet='db4', mode='periodization')
    return x_hat


def soft_threshold(x, beta):
    y = np.maximum(0, x-beta) - np.maximum(0, -x-beta)
    return y



# A_fun, AT_fun takes a vector (d,1) or (d,) as input
def solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, base_folder, show_img_progress=False, alpha=0.2, max_iter=100, solver_tol=1e-6):
    """ See Wang, Yu, Wotao Yin, and Jinshan Zeng. "Global convergence of ADMM in nonconvex nonsmooth optimization."
    arXiv preprint arXiv:1511.06324 (2015).
    It provides convergence condition: basically with large enough alpha, the program will converge. """

    #result_folder = '%s/iter-imgs' % base_folder
    #if not os.path.exists(result_folder):
        #os.makedirs(result_folder)

    obj_lss = np.zeros(max_iter)
    x_zs = np.zeros(max_iter)
    u_norms = np.zeros(max_iter)
    times = np.zeros(max_iter)

    ATy = AT_fun(y)
    x_shape = ATy.shape
    d = np.prod(x_shape)

    def A_cgs_fun(x):
        x = np.reshape(x, x_shape, order='F')
        y = AT_fun(A_fun(x)) + alpha * x
        return vec(y)
    A_cgs = LinearOperator((d,d), matvec=A_cgs_fun, dtype='float')

    def compute_p_inv_A(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs, vec(b), x0=vec(z0), tol=1e-3, maxiter=100)
        if info > 0:
            print 'cgs convergence to tolerance not achieved'
        elif info <0:
            print 'cgs gets illegal input or breakdown'
        z = np.reshape(z, x_shape, order='F')
        return z


    def A_cgs_fun_init(x):
        x = np.reshape(x, x_shape, order='F')
        y = AT_fun(A_fun(x))
        return vec(y)
    A_cgs_init = LinearOperator((d,d), matvec=A_cgs_fun_init, dtype='float')

    def compute_init(b, z0):
        (z,info) = sp.sparse.linalg.cgs(A_cgs_init, vec(b), x0=vec(z0), tol=1e-2)
        if info > 0:
            print 'cgs convergence to tolerance not achieved'
        elif info <0:
            print 'cgs gets illegal input or breakdown'
        z = np.reshape(z, x_shape, order='F')
        return z

    # initialize z and u
    z = compute_init(ATy, ATy)
    u = np.zeros(x_shape)


    plot_normalozer = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)


    start_time = timeit.default_timer()

    for iter in range(max_iter):

        # x-update
        net_input = z+u
        Wzu, wbook = wavelet_transform(net_input)
        q = soft_threshold(Wzu, lambda_l1/alpha)
        x = inverse_wavelet_transform(q, wbook, x_shape)
        x = np.reshape(x, x_shape)

        # z-update
        b = ATy + alpha * (x - u)
        z = compute_p_inv_A(b, z)

        # u-update
        u += z - x;

        if show_img_progress == True:

            fig = plt.figure('current_sol')
            plt.gcf().clear()
            fig.canvas.set_window_title('iter %d' % iter)
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(np.clip(x, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('x')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(np.clip(z, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('z')
            plt.subplot(1,3,3)
            plt.imshow(reshape_img_fun(np.clip(net_input, 0.0, 1.0)), interpolation='nearest', norm=plot_normalozer)
            plt.title('netin')
            plt.pause(0.00001)


        obj_ls = 0.5 * np.sum(np.square(y - A_fun(x)))
        x_z = np.sqrt(np.mean(np.square(x-z)))
        u_norm = np.sqrt(np.mean(np.square(u)))

        print 'iter = %d: obj_ls = %.3e  |x-z| = %.3e  u_norm = %.3e' % (iter, obj_ls, x_z, u_norm)


        obj_lss[iter] = obj_ls
        x_zs[iter] = x_z
        u_norms[iter] = u_norm
        times[iter] = timeit.default_timer() - start_time


        ## save images
        #filename = '%s/%d-x.jpg' % (result_folder, iter)
        #sp.misc.imsave(filename, sp.misc.imresize((reshape_img_fun(np.clip(x, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        #filename = '%s/%d-z.jpg' % (result_folder, iter)
        #sp.misc.imsave(filename, sp.misc.imresize((reshape_img_fun(np.clip(z, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        #filename = '%s/%d-u.jpg' % (result_folder, iter)
        #sp.misc.imsave(filename, sp.misc.imresize((reshape_img_fun(np.clip(u, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        #_ = raw_input('')

        if x_z < solver_tol:
            break

    infos = {'obj_lss': obj_lss, 'x_zs': x_zs, 'u_norms': u_norms,
             'times': times, 'alpha':alpha, 'lambda_l1':lambda_l1,
             'max_iter':max_iter, 'solver_tol':solver_tol}


    return (x, z, u, infos)

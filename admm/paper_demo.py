import sys
sys.path.append("../projector")
import main as model
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import math
import load_celeb as load_data
import os
import timeit
import matplotlib.pyplot as plt
import add_noise
import solver_paper as solver
import solver_l1 as solver_l1
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


def save_results(folder, infos, x, z, u):
    filename = '%s/infos.mat' % folder
    sp.io.savemat(filename, infos)
    filename = '%s/x.jpg' % folder
    sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(x, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
    filename = '%s/z.jpg' % folder
    sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(z, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
    filename = '%s/u.jpg' % folder
    sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(u, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))


# index of test images
idxs = np.random.randint(391854, size=1)

# result folder
clean_paper_results = 'clean_paper_results' 

# filename of the trained model. If using virtual batch normalization, 
# the popmean and popvariance need to be updated first via update_popmean.py!
iter = 49999
pretrained_folder = os.path.expanduser("../projector/model/imsize64_ratio0.010000_dis0.005000_latent0.000100_img0.001000_de1.000000_derate1.000000_dp1_gd1_softpos0.850000_wdcy_0.000000_seed0")
pretrained_model_file = '%s/update/model_iter-%d' % (pretrained_folder, iter)


for idx in idxs :
    print 'idx = %d --------' % idx

    np.random.seed(idx)

    img_size = (64,64,3)

    show_img_progress = False # whether the plot intermediate results (may slow down the process)
    run_ours = True           # whether the run the proposed method
    run_l1 = False            # whether the run the traditional wavelet sparsity method

    def load_image(filepath):
        img = sp.misc.imread(filepath)
        img = sp.misc.imresize(img, [64,64]).astype(float) / 255.0
        if len(img.shape) < 3:
            img = np.tile(img, [1,1,3])
        return img
    
    
    #def load_image(filepath): 
        #img = sp.misc.imread(filepath)
        ## <Note> In our original code used to generate the results in the paper, we mistakenly
        ## resize the image directly to the input dimension via
        ## img = sp.misc.imresize(img, [img_size[0], img_size[1]]).astype(float) / 255.0
        ## The following is the corrected version
        #img_shape = img.shape
        #min_edge = min(img_shape[0], img_shape[1])
        #min_resize_ratio = float(img_size[0]) / float(min_edge)
        #max_resize_ratio = min_resize_ratio * 2.0
        #resize_ratio = np.random.rand() * (max_resize_ratio - min_resize_ratio) + min_resize_ratio
    
        #img = sp.misc.imresize(img, resize_ratio).astype(float) / 255.0
        #crop_loc_row = np.random.randint(img.shape[0]-img_size[0]+1)
        #crop_loc_col = np.random.randint(img.shape[1]-img_size[1]+1)
        #if len(img.shape) == 3:
            #img = img[crop_loc_row:crop_loc_row+img_size[0], crop_loc_col:crop_loc_col+img_size[1],:]
        #else:
            #img = img[crop_loc_row:crop_loc_row+img_size[0], crop_loc_col:crop_loc_col+img_size[1]]        
        #if len(img.shape) < 3:
            #img = np.tile(img, [1,1,3])
        #return img        

    def solve_denoising_dropping(ori_img, denoiser, reshape_img_fun, drop_prob=0.3,
                                 noise_mean=0, noise_std=0.1,
                                 alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask) = problem.setup(x_shape, drop_prob=drop_prob)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('denoise')
            plt.gcf().clear()
            fig.canvas.set_window_title('denoise')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'drop_prob': drop_prob, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}

        # save the problem
        base_folder = '%s/denoise_ratio%.2f_std%.2f' % (result_folder, drop_prob, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        if run_ours:
            # ours
            folder = '%s/ours_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol)
            save_results(folder, infos, x, z, u)

        if run_l1:
            # wavelet l1
            folder = '%s/l1_lambdal1%f_alpha%f' % (base_folder, lambda_l1, alpha_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_l1.solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, folder,
                                        show_img_progress=show_img_progress, alpha=alpha_l1,
                                        max_iter=max_iter_l1, solver_tol=solver_tol_l1)
            save_results(folder, infos, x, z, u)

        z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
        img = Image.fromarray( sp.misc.imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype(font='tnr.ttf', size=50)
        #draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
        filename = '%s/z.jpg' % folder
        img.save(filename)

    def solve_inpaint_center(ori_img, denoiser, reshape_img_fun, box_size=1,
                            noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint_center as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask) = problem.setup(x_shape, box_size=box_size)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('inpaint_center')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint_center')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'box_size': box_size, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}

        # save the problem
        base_folder = '%s/inpaintcenter_bs%d_std%.2f' % (result_folder, box_size, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

        if run_ours:
            # ours
            folder = '%s/ours_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol)
            save_results(folder, infos, x, z, u)

        if run_l1:
            # wavelet l1
            folder = '%s/l1_lambdal1%f_alpha%f' % (base_folder, lambda_l1, alpha_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_l1.solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, folder,
                                            show_img_progress=show_img_progress, alpha=alpha_l1,
                                            max_iter=max_iter_l1, solver_tol=solver_tol_l1)
            save_results(folder, infos, x, z, u)

        z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
        img = Image.fromarray( sp.misc.imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype(font='tnr.ttf', size=50)
        #draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
        filename = '%s/z.jpg' % folder
        img.save(filename)

    def solve_inpaint_block(ori_img, denoiser, reshape_img_fun, box_size=1, total_box=1,
                            noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2):
        import inpaint_block as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, mask) = problem.setup(x_shape, box_size=box_size, total_box=total_box)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        if show_img_progress:
            fig = plt.figure('inpaint')
            plt.gcf().clear()
            fig.canvas.set_window_title('inpaint')
            plt.subplot(1,2,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,2,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.pause(0.00001)


        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'mask': mask, 'box_size': box_size,
                'total_box': total_box, 'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}


        # save the problem
        base_folder = '%s/inpaint_bs%d_tb%d_std%.2f' % (result_folder, box_size, total_box, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))


        if run_ours:
            # ours
            folder = '%s/ours_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol)
            save_results(folder, infos, x, z, u)

        if run_l1:
            # wavelet l1
            folder = '%s/l1_lambdal1%f_alpha%f' % (base_folder, lambda_l1, alpha_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_l1.solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, folder,
                                            show_img_progress=show_img_progress, alpha=alpha_l1,
                                            max_iter=max_iter_l1, solver_tol=solver_tol_l1)
            save_results(folder, infos, x, z, u)

        z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
        img = Image.fromarray( sp.misc.imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype(font='tnr.ttf', size=50)
        #draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
        filename = '%s/z.jpg' % folder
        img.save(filename)


    def solve_superres(ori_img, denoiser, reshape_img_fun, resize_ratio=0.5,
                       noise_mean=0, noise_std=0.,
                       alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2):
        import superres as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun) = problem.setup(x_shape, resize_ratio=resize_ratio)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        bicubic_img = sp.misc.imresize(y[0], [ori_img.shape[1], ori_img.shape[2]], interp='bicubic')
        if show_img_progress:
            fig = plt.figure('superres')
            plt.gcf().clear()
            fig.canvas.set_window_title('superres')
            plt.subplot(1,3,1)
            plt.imshow(reshape_img_fun(ori_img), interpolation='nearest')
            plt.title('ori_img')
            plt.subplot(1,3,2)
            plt.imshow(reshape_img_fun(y), interpolation='nearest')
            plt.title('y')
            plt.subplot(1,3,3)
            plt.imshow(np.clip(bicubic_img,0,255), interpolation='nearest')
            plt.title('bicubic')
            plt.pause(0.00001)

        bicubic_img = bicubic_img.astype(float) / 255.0
        l2_dis = np.mean(np.square(ori_img[0] - bicubic_img))

        print 'bicubic err = %f' % (l2_dis)


        info = {'ori_img': ori_img, 'y': y, 'bicubic': bicubic_img, 'noise': noise, 'resize_ratio': resize_ratio,
                'noise_std': noise_std,
                'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}

        # save the problem
        base_folder = '%s/superres_ratio%.2f_std%.2f' % (result_folder, resize_ratio, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/y.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(y, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/ori_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))
        filename = '%s/bicubic_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((bicubic_img*255).astype(np.uint8), 4.0, interp='nearest'))

        if run_ours:
            # ours
            folder = '%s/ours_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol)
            save_results(folder, infos, x, z, u)

        if run_l1:
            # wavelet l1
            folder = '%s/l1_lambdal1%f_alpha%f' % (base_folder, lambda_l1, alpha_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_l1.solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, folder,
                                            show_img_progress=show_img_progress, alpha=alpha_l1,
                                            max_iter=max_iter_l1, solver_tol=solver_tol_l1)
            save_results(folder, infos, x, z, u)

        z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
        img = Image.fromarray( sp.misc.imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype(font='tnr.ttf', size=50)
        #draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
        filename = '%s/z.jpg' % folder
        img.save(filename)

    def solve_cs(ori_img, denoiser, reshape_img_fun, compress_ratio,
                            noise_mean=0, noise_std=0.,
                            alpha=0.3, lambda_l1=0.1, max_iter=100, solver_tol=1e-2):
        import cs as problem
        x_shape = ori_img.shape
        (A_fun, AT_fun, A) = problem.setup(x_shape, compress_ratio)
        y, noise = add_noise.exe(A_fun(ori_img), noise_mean=noise_mean, noise_std=noise_std)

        info = {'ori_img': ori_img, 'y': y, 'noise': noise, 'compress_ratio': compress_ratio,
                    'noise_std': noise_std,
                    'alpha': alpha, 'max_iter': max_iter, 'solver_tol': solver_tol, 'lambda_l1': lambda_l1}

        # save the problem
        base_folder = '%s/cs_ratio%.2f_std%.2f' % (result_folder, compress_ratio, noise_std)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        filename = '%s/settings.mat' % base_folder
        sp.io.savemat(filename, info)
        filename = '%s/ori_img.jpg' % base_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))


        if run_ours:
            # ours
            folder = '%s/ours_alpha%f' % (base_folder, alpha)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver.solve(y, A_fun, AT_fun, denoiser, reshape_img_fun, folder,
                                         show_img_progress=show_img_progress, alpha=alpha,
                                       max_iter=max_iter, solver_tol=solver_tol)
            save_results(folder, infos, x, z, u)

        if run_l1:
            # wavelet l1
            folder = '%s/l1_lambdal1%f_alpha%f' % (base_folder, lambda_l1, alpha_l1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            (x, z, u, infos) = solver_l1.solve(y, A_fun, AT_fun, lambda_l1, reshape_img_fun, folder,
                                            show_img_progress=show_img_progress, alpha=alpha_l1,
                                            max_iter=max_iter_l1, solver_tol=solver_tol_l1)
            save_results(folder, infos, x, z, u)

        z1 = reshape_img(np.clip(z, 0.0, 1.0)) 
        ori_img1 = reshape_img(np.clip(ori_img, 0.0, 1.0)) 
        psnr = 10*np.log10( 1.0 /np.linalg.norm(z1-ori_img1)**2*np.prod(z1.shape))   
        img = Image.fromarray( sp.misc.imresize(np.uint8(z1*255), 4.0, interp='nearest' ) )
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype(font='tnr.ttf', size=50)
        #draw.text((135, 200), "%.2f"%psnr, (255,255,255), font=font)
        filename = '%s/z.jpg' % folder
        img.save(filename)
        
    def reshape_img(img):
        return img[0]


    if run_ours:
        # setup the variables in the session
        n_reference = 0
        batch_size = n_reference + 1
        images_tf = tf.placeholder( tf.float32, [batch_size, img_size[0], img_size[1], img_size[2]], name="images")
        is_train = False
        proj, latent = model.build_projection_model(images_tf, is_train, n_reference, use_bias=True, reuse=None)

        with tf.variable_scope("PROJ") as scope:
            scope.reuse_variables()

    # load the dataset


    print 'loading data...'
    testset_filelist = load_data.load_testset_path_list()
    total_test = len(testset_filelist)
    print 'total test = %d' % total_test

    # We create a session to use the graph and restore the variables
    if run_ours:
        print 'loading model...'
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, pretrained_model_file)
        #print(sess.run(tf.global_variables()))
        print 'finished reload.'

    # define denoiser
    def denoise(x):
        x_shape = x.shape
        x = np.reshape(x, [1, img_size[0], img_size[1], img_size[2]], order='F')
        x = (x - 0.5) * 2.0

        y = sess.run(proj, feed_dict={images_tf: x})
        y = (y / 2.0) + 0.5
        return np.reshape(y, x_shape)


    def denoise_batch(x):
        x_shape = x.shape

        ys = np.zeros(x_shape)
        for i in range(x_shape[0]):
            ys[i] = denoise(x[i])

        return ys

    img =load_image(testset_filelist[idx])

    ori_img = np.reshape(img, [1, img_size[0],img_size[1],img_size[2]], order='F')

    result_folder = '%s/%d' % (clean_paper_results,idx)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    if run_ours:
        direct_img = denoise(ori_img)
    if show_img_progress:
        plt.figure('original')
        img_plot = plt.imshow(reshape_img(ori_img))
        plt.pause(0.001)

        plt.figure('direct')
        img_plot = plt.imshow((reshape_img(np.clip(denoise(direct_img),0.0,1.0))*255).astype(np.uint8))
        plt.pause(0.001)


    filename = '%s/ori_img.jpg' % result_folder
    sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(ori_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))

    if run_ours:
        filename = '%s/direct_img.jpg' % result_folder
        sp.misc.imsave(filename, sp.misc.imresize((reshape_img(np.clip(direct_img, 0.0, 1.0))*255).astype(np.uint8), 4.0, interp='nearest'))


    ##############################################################################################
    ##### super resolution
    print 'super resolution'

    #set parameters
    alpha = 0.5 # 1.0
    max_iter = 30
    solver_tol = 2e-3

    alpha_l1 = 0.3
    lambda_l1 = 0.05
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4

    resize_ratio = 0.5
    noise_std = 0.0
    results = solve_superres(ori_img, denoise, reshape_img, resize_ratio=resize_ratio,
                             noise_std=noise_std,
                             alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol)

    #################################################################################################
    ##### compressive sensing
    print 'compressive sensing'

    #set parameters
    alpha = 0.3
    max_iter = 300
    solver_tol = 3e-3

    alpha_l1 = 0.3
    lambda_l1 = 0.05
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4

    compress_ratio = 0.1
    noise_std = 0.0
    results = solve_cs(ori_img, denoise, reshape_img, compress_ratio=compress_ratio,
                       noise_std=noise_std,
                       alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol)

    ############################################################################################
    #### denoising

    print 'denoising'

    # set parameter
    alpha = 0.3
    max_iter = 300
    solver_tol = 3e-3

    alpha_l1 = 0.3
    lambda_l1 = 0.05
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4

    drop_prob = 0.5
    noise_std = 0.1

    results = solve_denoising_dropping(ori_img, denoise, reshape_img, drop_prob=drop_prob,
                                       noise_mean=0, noise_std=noise_std,
                                       alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol)

    ##########################################################################################
    ## inpaint block

    print 'inpaint block'

    # set parameter
    alpha = 0.3
    max_iter = 300
    solver_tol = 1e-4

    alpha_l1 = 0.3
    lambda_l1 = 0.03
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4

    box_size = int(0.1 * ori_img.shape[1])
    noise_std = 0.0
    total_box = 10
    results = solve_inpaint_block(ori_img, denoise, reshape_img, box_size=box_size, total_box=total_box,
                                  noise_std=noise_std,
                                  alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol)
    
    ############################################################################################
    ### inpaint center
    print 'inpaint center'
   
    alpha = 0.2 
    max_iter = 300
    solver_tol = 1e-5
    alpha_update_ratio = 1.0
   
    alpha_l1 = 0.3
    lambda_l1 = 0.05
    max_iter_l1 = 1000
    solver_tol_l1 = 1e-4
   
    box_size = int(0.3 * ori_img.shape[1])
    noise_std = 0.0
    results = solve_inpaint_center(ori_img, denoise, reshape_img, box_size=box_size,
    noise_std=noise_std,
    alpha=alpha, lambda_l1=lambda_l1, max_iter=max_iter, solver_tol=solver_tol)

    if run_ours:
        tf.reset_default_graph()

raw_input("Press Enter to end...")

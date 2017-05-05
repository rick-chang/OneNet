import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import scipy.misc
import scipy.ndimage
import layers_nearest_2 as layers
import load_celeb as load_dataset
import os
import os.path
import timeit
from multiprocessing import Process, Queue
import argparse
from smooth_stream import SmoothStream
from noise import add_noise
import sys


def build_classifier_model_imagespace(image, is_train, n_reference, reuse=None):
    """
    Build the graph for the classifier in the image space
    """

    channel_compress_ratio = 4

    with tf.variable_scope('DIS', reuse=reuse):

        with tf.variable_scope('IMG'):
            ## image space D
            # 1
            conv1 = layers.new_conv_layer(image, [4,4,3,64], stride=1, name="conv1" ) #64

            # 2
            nBlocks = 3
            module2 = layers.add_bottleneck_module(conv1, is_train, nBlocks, n_reference, channel_compress_ratio=channel_compress_ratio, name='module2') # 32

            # 3
            nBlocks = 4
            module3 = layers.add_bottleneck_module(module2, is_train, nBlocks, n_reference, channel_compress_ratio=channel_compress_ratio, name='module3') # 16

            # 4
            nBlocks = 6
            module4 = layers.add_bottleneck_module(module3, is_train, nBlocks, n_reference, channel_compress_ratio=channel_compress_ratio, name='module4') # 8

            # 5
            nBlocks = 3
            module5 = layers.add_bottleneck_module(module4, is_train, nBlocks, n_reference, channel_compress_ratio=channel_compress_ratio, name='module5') # 4
            bn_module5 = tf.nn.elu(layers.batchnorm(module5, is_train, n_reference, name='bn_module5'))

            (dis, last_w) = layers.new_fc_layer(bn_module5, output_size=1, name='dis')

    return dis[:,0], last_w



def build_classifier_model_latentspace(latent, is_train, n_reference, reuse=None):
    """
    Build the graph for the classifier in the latent space
    """

    channel_compress_ratio = 4

    with tf.variable_scope('DIS', reuse=reuse):

        with tf.variable_scope('LATENT'):

            out = layers.bottleneck(latent, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=1, name='block0') # 8*8*4096
            out = layers.bottleneck(out, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=1, name='block1') # 8*8*4096
            out = layers.bottleneck(out, is_train, n_reference, channel_compress_ratio=channel_compress_ratio, stride=1, name='block2') # 8*8*4096

            output_channel = out.get_shape().as_list()[-1]
            out = layers.bottleneck_flexible(out, is_train, output_channel, n_reference, channel_compress_ratio=4, stride=2, name='block3') # 4*4*4096
            out = layers.bottleneck(out, is_train, n_reference, channel_compress_ratio=4, stride=1, name='block4') # 4*4*4096
            out = layers.bottleneck(out, is_train, n_reference, channel_compress_ratio=4, stride=1, name='block5') # 4*4*4096

            bn1 = tf.nn.elu(layers.batchnorm(out, is_train, n_reference, name='bn1'))
            (dis, last_w) = layers.new_fc_layer(bn1, output_size=1, name='dis')

    return dis[:,0], last_w


def build_projection_model(images, is_train, n_reference, use_bias=True, reuse=None):
    """
    Build the graph for the projection network, which shares the architecture of a typical autoencoder.
    To improve contextual awareness, we add a channel-wise fully-connected layer followed by a 2-by-2
    convolution layer at the middle.
    """
    channel_compress_ratio = 4
    dim_latent = 1024

    with tf.variable_scope('PROJ', reuse=reuse):

        with tf.variable_scope('ENCODE'):
            conv0 = layers.new_conv_layer(images, [4,4,3,64], stride=1, bias=use_bias, name="conv0" ) #64
            bn0 = tf.nn.elu(layers.batchnorm(conv0, is_train, n_reference, name='bn0'))
            conv1 = layers.new_conv_layer(bn0, [4,4,64,128], stride=1, bias=use_bias, name="conv1" ) #64
            bn1 = tf.nn.elu(layers.batchnorm(conv1, is_train, n_reference, name='bn1'))
            conv2 = layers.new_conv_layer(bn1, [4,4,128,256], stride=2, bias=use_bias, name="conv2") #32
            bn2 = tf.nn.elu(layers.batchnorm(conv2, is_train, n_reference, name='bn2'))
            conv3 = layers.new_conv_layer(bn2, [4,4,256,512], stride=2, bias=use_bias, name="conv3") #16
            bn3 = tf.nn.elu(layers.batchnorm(conv3, is_train, n_reference, name='bn3'))
            conv4 = layers.new_conv_layer(bn3, [4,4,512,dim_latent], stride=2, bias=use_bias, name="conv4") #8
            bn4 = tf.nn.elu(layers.batchnorm(conv4, is_train, n_reference, name='bn4'))
            fc5 = layers.channel_wise_fc_layer(bn4, 'fc5', bias=False)
            fc5_conv = layers.new_conv_layer(fc5, [2,2,dim_latent, dim_latent], stride=1, bias=use_bias, name="conv_fc")
            latent = tf.nn.elu(layers.batchnorm(fc5_conv, is_train, n_reference, name='latent'))


        deconv3 = layers.new_deconv_layer( latent, [4,4,512,dim_latent], conv3.get_shape().as_list(), stride=2, bias=use_bias, name="deconv3")
        debn3 = tf.nn.elu(layers.batchnorm(deconv3, is_train, n_reference, name='debn3'))
        deconv2 = layers.new_deconv_layer( debn3, [4,4,256,512], conv2.get_shape().as_list(), stride=2, bias=use_bias, name="deconv2")
        debn2 = tf.nn.elu(layers.batchnorm(deconv2, is_train, n_reference, name='debn2'))
        deconv1 = layers.new_deconv_layer( debn2, [4,4,128,256], conv1.get_shape().as_list(), stride=2, bias=use_bias, name="deconv1")
        debn1 = tf.nn.elu(layers.batchnorm(deconv1, is_train, n_reference, name='debn1'))
        deconv0 = layers.new_deconv_layer( debn1, [4,4,64,128], conv0.get_shape().as_list(), stride=1, bias=use_bias, name="deconv0")
        debn0 = tf.nn.elu(layers.batchnorm(deconv0, is_train, n_reference, name='debn0'))
        proj_ori = layers.new_deconv_layer( debn0, [4,4,3,64], images.get_shape().as_list(), stride=1, bias=use_bias, name="recon")
        proj = proj_ori

    return proj, latent



if __name__ == '__main__':

    ### parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', default=None, help='Where to store samples and models')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--img_size', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--n_reference', type=int, default=32, help='the size of reference batch')
    parser.add_argument('--Dperiod', type=int, default=1, help='number of continuous D update')
    parser.add_argument('--Gperiod', type=int, default=1, help='number of continuous G update')

    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--pretrained_iter', type=int, default=0, help='iter of the pretrained model, if 0 then not using')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')

    parser.add_argument('--learning_rate_val_proj', type=float, default=0.002, help='learning rate, default=0.002')
    parser.add_argument('--learning_rate_val_dis', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--weight_decay_rate', type=float, default=0.00001, help='weight decay rate, default=0.00000')
    parser.add_argument('--clamp_weight', type=int, default=1)
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)

    parser.add_argument('--use_spatially_varying_uniform_on_top', type=int, default=1, help='Whether to multiply the gaussian noise with a uniform noise map to avoid overfitting')

    parser.add_argument('--continuous_noise', type=int, default=1, help='whether to use continuous noise_std ')
    parser.add_argument('--noise_std', type=float, default=1.2, help='std of the added noise, default = 1.2')

    parser.add_argument('--uniform_noise_max', type=float, default=3.464, help='The range of the uniform noise, default = 3.464 to make overall std remain unchange')
    parser.add_argument('--min_spatially_continuous_noise_factor', type=float, default=0.01, help='The lower the value, the higher the possibility the varying of the noise be more continuous')
    parser.add_argument('--max_spatially_continuous_noise_factor', type=float, default=0.5, help='The upper the value, the higher the possibility the varying of the noise be more continuous')
    parser.add_argument('--adam_beta1_d', type=float, default=0.9, help='beta1 of adam for the critic, default = 0.9')
    parser.add_argument('--adam_beta2_d', type=float, default=0.999, help='beta2 of adam for the critic, default = 0.999')
    parser.add_argument('--adam_eps_d', type=float, default=1e-8, help='eps of adam for the critic, default = 1e-8')
    parser.add_argument('--adam_beta1_g', type=float, default=0.9, help='beta1 of adam for the projector, default = 0.9')
    parser.add_argument('--adam_beta2_g', type=float, default=0.999, help='beta2 of adam for the projector, default = 0.999')
    parser.add_argument('--adam_eps_g', type=float, default=1e-5, help='eps of adam for the projector, default = 1e-8')

    parser.add_argument('--use_tensorboard', type=int, default=1, help='whether to use tensorboard')
    parser.add_argument('--tensorboard_period', type=int, default=1, help='how often to write to tensorboard')
    parser.add_argument('--output_img', type=int, default=0, help='whether to output images, (also act as the number of images to output)')
    parser.add_argument('--output_img_period', type=int, default=100, help='how often to output images')

    parser.add_argument('--clip_input', type=int, default=0, help='clip the input to the network')
    parser.add_argument('--clip_input_bound', type=float, default=2.0, help='the maximum of input')

    parser.add_argument('--lambda_ratio', type=float, default=1e-2, help='the weight ratio in the objective function of true images to fake images, default 1e-2')
    parser.add_argument('--lambda_l2', type=float, default=5e-3, help='lambda of l2 loss, default = 5e-3')
    parser.add_argument('--lambda_latent', type=float, default=1e-4, help='lambda of latent adv loss, default = 1e-4')
    parser.add_argument('--lambda_img', type=float, default=1e-3, help='lambda of img adv loss, default = 1e-3')
    parser.add_argument('--lambda_de', type=float, default=1.0, help='lambda of the denoising autoencoder, default = 1.0')
    parser.add_argument('--de_decay_rate', type=float, default=1.0, help='the rate lambda_de decays, default = 1.0')

    parser.add_argument('--one_sided_label_smooth', type=float, default=0.85, help='the positive label for one-sided, default = 0.85')

    opt = parser.parse_args()
    print(opt)

    ### parameters ###
    n_epochs = int(opt.n_epochs)
    learning_rate_val_dis = float(opt.learning_rate_val_dis)
    learning_rate_val_proj = float(opt.learning_rate_val_proj)
    learning_rate_val_proj_max = learning_rate_val_proj
    learning_rate_val_proj_current = learning_rate_val_proj
    weight_decay_rate =  float(opt.weight_decay_rate)
    batch_size = int(opt.batch_size)

    std = float(opt.noise_std)
    continuous_noise = int(opt.continuous_noise)

    use_spatially_varying_uniform_on_top = int(opt.use_spatially_varying_uniform_on_top)
    uniform_noise_max = float(opt.uniform_noise_max)
    min_spatially_continuous_noise_factor = float(opt.min_spatially_continuous_noise_factor)
    max_spatially_continuous_noise_factor = float(opt.max_spatially_continuous_noise_factor)


    img_size = int(opt.img_size)
    Dperiod = int(opt.Dperiod)
    Gperiod = int(opt.Gperiod)

    clamp_weight = int(opt.clamp_weight)
    clamp_lower = float(opt.clamp_lower)
    clamp_upper = float(opt.clamp_upper)

    random_seed = int(opt.random_seed)

    adam_beta1_d = float(opt.adam_beta1_d)
    adam_beta2_d = float(opt.adam_beta2_d)
    adam_eps_d = float(opt.adam_eps_d)

    adam_beta1_g = float(opt.adam_beta1_g)
    adam_beta2_g = float(opt.adam_beta2_g)
    adam_eps_g = float(opt.adam_eps_g)

    use_tensorboard = int(opt.use_tensorboard)
    tensorboard_period = int(opt.tensorboard_period)
    output_img = int(opt.output_img)
    output_img_period = int(opt.output_img_period)

    clip_input = int(opt.clip_input)
    clip_input_bound = float(opt.clip_input_bound)

    lambda_ratio = float(opt.lambda_ratio)
    lambda_l2 = float(opt.lambda_l2)
    lambda_latent = float(opt.lambda_latent)
    lambda_img = float(opt.lambda_img)
    lambda_de = float(opt.lambda_de)
    de_decay_rate = float(opt.de_decay_rate)

    one_sided_label_smooth = float(opt.one_sided_label_smooth)

    n_reference = int(opt.n_reference)
    inst_size = batch_size - n_reference


    base_folder = opt.base_folder

    if base_folder == None:
        base_folder = 'model'

    base_folder = '%s/imsize%d_ratio%f_dis%f_latent%f_img%f_de%f_derate%f_dp%d_gd%d_softpos%f_wdcy_%f_seed%d' % (
        base_folder, img_size, lambda_ratio, lambda_l2, lambda_latent, lambda_img,
        lambda_de, de_decay_rate,
        Dperiod, Gperiod,
        one_sided_label_smooth,
        weight_decay_rate,
        random_seed
    )

    model_path = '%s/model' % (base_folder)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    epoch_path = '%s/epoch' % (base_folder)
    if not os.path.exists(epoch_path):
        os.makedirs(epoch_path)

    init_path = '%s/init' % (base_folder)
    if not os.path.exists(init_path):
        os.makedirs(init_path)

    img_path = '%s/image' % (base_folder)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    logs_base = '/tmp/tensorflow_logs'
    logs_path = '%s/%s' % (logs_base, base_folder)

    # write configurations to a file
    filename = '%s/configurations.txt' % (base_folder)
    f = open( filename, 'a' )
    f.write( repr(opt) + '\n' )
    f.close()

    pretrained_iter = int(opt.pretrained_iter)
    use_pretrain = pretrained_iter > 0
    pretrained_model_file = '%s/model_iter-%d' % (model_path, pretrained_iter)

    tf.set_random_seed(random_seed)


    ### load the dataset ###

    def read_file_cpu(trainset, queue, batch_size, num_prepare, rseed=None):
        local_random = np.random.RandomState(rseed)

        n_train = len(trainset)
        trainset_index = local_random.permutation(n_train)
        idx = 0
        while True:
            # read in data if the queue is too short
            while queue.full() == False:
                batch = np.zeros([batch_size, img_size, img_size, 3])
                noisy_batch = np.zeros([batch_size, img_size, img_size, 3])
                for i in range(batch_size):
                    image_path = trainset[trainset_index[idx+i]]
                    img = sp.misc.imread(image_path)
                    # <Note> In our original code used to generate the results in the paper, we directly
                    # resize the image directly to the input dimension via (for both ms-celeb-1m and imagenet)
                    img = sp.misc.imresize(img, [img_size, img_size]).astype(float) / 255.0
                    
                    # The following code crops random-sized patches (may be useful for imagenet)
                    #img_shape = img.shape
                    #min_edge = min(img_shape[0], img_shape[1])
                    #min_resize_ratio = float(img_size) / float(min_edge)
                    #max_resize_ratio = min_resize_ratio * 2.0
                    #resize_ratio = local_random.rand() * (max_resize_ratio - min_resize_ratio) + min_resize_ratio

                    #img = sp.misc.imresize(img, resize_ratio).astype(float) / 255.0
                    #crop_loc_row = local_random.randint(img.shape[0]-img_size+1)
                    #crop_loc_col = local_random.randint(img.shape[1]-img_size+1)
                    #if len(img.shape) == 3:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size,:]
                    #else:
                        #img = img[crop_loc_row:crop_loc_row+img_size, crop_loc_col:crop_loc_col+img_size]

                    if np.prod(img.shape) == 0:
                        img = np.zeros([img_size, img_size, 3])

                    if len(img.shape) < 3:
                        img = np.expand_dims(img, axis=2)
                        img = np.tile(img, [1,1,3])

                    ## random flip
                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[-1:None:-1,:,:]

                    #flip_prob = local_random.rand()
                    #if flip_prob < 0.5:
                        #img = img[:,-1:None:-1,:]

                    # add noise to img
                    noisy_img = add_noise(img, local_random,
                            std=std,
                            uniform_max=uniform_noise_max,
                            min_spatially_continuous_noise_factor=min_spatially_continuous_noise_factor,
                            max_spatially_continuous_noise_factor=max_spatially_continuous_noise_factor,
                            continuous_noise=continuous_noise,
                            use_spatially_varying_uniform_on_top=use_spatially_varying_uniform_on_top,
                            clip_input=clip_input, clip_input_bound=clip_input_bound
                            )

                    batch[i] = img
                    noisy_batch[i] = noisy_img

                batch *= 2.0
                batch -= 1.0
                noisy_batch *= 2.0
                noisy_batch -= 1.0

                if clip_input > 0:
                    batch = np.clip(batch, a_min=-clip_input_bound, a_max=clip_input_bound)
                    noisy_batch = np.clip(noisy_batch, a_min=-clip_input_bound, a_max=clip_input_bound)

                queue.put([batch, noisy_batch]) # block until free slot is available

                idx += batch_size
                if idx > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
                    trainset_index = local_random.permutation(n_train)
                    idx = 0

    def create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs):
        """
        create threads to read the images from hard drive and perturb them
        """
        for n_read in range(n_thread):
            seed = np.random.randint(1e8)
            instance_size = batch_size - n_reference
            if instance_size < 1:
                print 'ERROR: batch_size < n_reference + 1'
            train_proc = Process(target=read_file_cpu, args=(trainset, train_queue, instance_size, num_prepare, seed))
            train_proc.daemon = True
            train_proc.start()
            train_procs.append(train_proc)

    def terminate_train_procs(train_procs):
        """
        terminate the threads to force garbage collection and free memory
        """
        for procs in train_procs:
            procs.terminate()


    trainset = load_dataset.load_trainset_path_list()
    total_train = len(trainset)
    print 'total train = %d' % (total_train)


    print "create reference batch..."
    n_thread = 1
    num_prepare = 1
    reference_queue = Queue(num_prepare)
    ref_seed = 1085 # the random seed particularly for creating the reference batch
    ref_proc = Process(target=read_file_cpu, args=(trainset, reference_queue, n_reference, num_prepare, ref_seed))
    ref_proc.daemon = True
    ref_proc.start()

    _, ref_batch = reference_queue.get()

    ref_proc.terminate()
    del ref_proc
    del reference_queue

    # save reference to a mat file
    ref_file = '%s/ref_batch_%d.mat' % (base_folder, n_reference)
    sp.io.savemat(ref_file, {'ref_batch': ref_batch})
    print 'ref_batch saved.'

    def np_combine_batch(inst,ref):
        out = np.concatenate([inst,ref], axis=0)
        return out
    def get_inst(batch):
        return batch[0:inst_size]


    print "loading data..."

    n_thread = 16
    num_prepare = 20
    print 'total train = %d' % (total_train)
    train_queue = Queue(num_prepare+1)
    train_procs = []
    create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs)


    ### set up the graph
    # images
    images_tf = tf.placeholder( tf.float32, [batch_size, img_size, img_size, 3], name="images_tf")
    noisy_image_tf = tf.placeholder( tf.float32, [batch_size, img_size, img_size, 3], name="noisy_image_tf")

    # lambdas
    lambda_ratio_tf = tf.placeholder( tf.float32, [], name='lambda_ratio_tf')
    lambda_l2_tf = tf.placeholder( tf.float32, [], name='lambda_l2_tf')
    lambda_latent_tf = tf.placeholder( tf.float32, [], name='lambda_latent_tf')
    lambda_img_tf = tf.placeholder( tf.float32, [], name='lambda_img')
    lambda_de_tf = tf.placeholder( tf.float32, [], name='lambda_de')

    is_train = True
    learning_rate_dis = tf.placeholder( tf.float32, [], name='learning_rate_dis')
    learning_rate_proj = tf.placeholder( tf.float32, [], name='learning_rate_proj')
    adam_beta1_d_tf = tf.placeholder( tf.float32, [], name='adam_beta1_d_tf')
    adam_beta1_g_tf = tf.placeholder( tf.float32, [], name='adam_beta1_g_tf')


    images_dataset = images_tf

    # build autoencoder
    projection_x_all, latent_x_all = build_projection_model(images_dataset, is_train, n_reference)
    projection_x = get_inst(projection_x_all)
    latent_x = get_inst(latent_x_all)

    projection_z_all, latent_z_all = build_projection_model(noisy_image_tf, is_train, n_reference, reuse=True)
    projection_z = get_inst(projection_z_all)
    latent_z = get_inst(latent_z_all)

    # build the discriminator
    # image space
    adversarial_truex_all, _ = build_classifier_model_imagespace(images_dataset, is_train, n_reference)
    adversarial_truex = get_inst(adversarial_truex_all)

    adversarial_projx_all, _ = build_classifier_model_imagespace(projection_x, is_train, n_reference, reuse=True)
    adversarial_projx = get_inst(adversarial_projx_all)

    adversarial_projz_all, _ = build_classifier_model_imagespace(projection_z, is_train, n_reference, reuse=True)
    adversarial_projz = get_inst(adversarial_projz_all)

    # latent space
    if lambda_latent > 0:
        adversarial_latentx_all, _ = build_classifier_model_latentspace(latent_x, is_train, n_reference)
        adversarial_latentz_all, _ = build_classifier_model_latentspace(latent_z, is_train, n_reference, reuse=True)
    else:
        adversarial_latentx_all = tf.zeros([batch_size])
        adversarial_latentz_all = tf.zeros([batch_size])

    adversarial_latentx = get_inst(adversarial_latentx_all)
    adversarial_latentz = get_inst(adversarial_latentz_all)


    # update_op for batch_norm moving average
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
    else:
        print 'something is wrong!'
    
    # if we are using virtual batch normalization, we do not need to calculate the population mean and variance
    if n_reference > 0:
        updates = tf.zeros([1])

    # set up the loss for D
    pos_labels = tf.ones([inst_size],1)
    soft_pos_labels = pos_labels * one_sided_label_smooth
    neg_labels = tf.zeros([tf.shape(adversarial_latentz)[0]],1)

    loss_adv_D_pos_latent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_latentx, soft_pos_labels))
    loss_adv_D_neg_latent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_latentz, neg_labels))

    loss_adv_D_latent = lambda_ratio_tf*loss_adv_D_pos_latent + (1-lambda_ratio_tf)*loss_adv_D_neg_latent

    loss_adv_D_pos_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_truex, soft_pos_labels))
    loss_adv_D_neg_img = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_projx, neg_labels))
    loss_adv_D_neg_imgz = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adversarial_projz, neg_labels))

    loss_adv_D_img = (loss_adv_D_pos_img + lambda_ratio_tf*loss_adv_D_neg_img + (1-lambda_ratio_tf)*loss_adv_D_neg_imgz ) * 0.5  # currently not using loss_adv_D_neg_imgz

    est_labels_latent = tf.to_float(tf.concat(0,
            [tf.greater_equal(adversarial_latentx,0.0),
             tf.less(adversarial_latentz,0.0),
             ] ))
    accuracy_latent = tf.reduce_mean(est_labels_latent, name='accuracy_latent')

    est_labels_img = tf.to_float(tf.concat(0,
            [tf.greater_equal(adversarial_truex,0.0),
             tf.less(adversarial_projx,0.0),
             ] ))
    accuracy_img = tf.reduce_mean(est_labels_img, name='accuracy_img')


    # set up the loss for autoencoder
    loss_proj = tf.reduce_mean(tf.square(projection_z - get_inst(noisy_image_tf)  )  )
    loss_recon = tf.reduce_mean(tf.square(projection_x - get_inst(images_dataset) ))
    loss_recon_z = tf.reduce_mean(tf.square(projection_z - get_inst(images_dataset) ))

    labels_G = pos_labels # flip label when training G


    # latent
    loss_adv_G_latent = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_latentz, labels_G))

    # imagespace
    loss_adv_G_imgx = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_projx, labels_G))
    loss_adv_G_imgz = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_projz, labels_G))
    loss_adv_G_img = lambda_ratio_tf*loss_adv_G_imgx + (1-lambda_ratio_tf)*loss_adv_G_imgz

    loss_adv_G = lambda_latent_tf * loss_adv_G_latent +  lambda_img_tf * loss_adv_G_img

    loss_adv_D = lambda_latent_tf * loss_adv_D_latent + lambda_img*loss_adv_D_img
    loss_G = loss_adv_G + lambda_l2_tf * (lambda_ratio_tf * loss_recon + (1-lambda_ratio_tf)*loss_proj )
    # train with a denoising autoencoder weight first
    loss_G += lambda_de_tf * loss_recon_z

    var_D = filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables())
    W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

    var_G = filter( lambda x: x.name.startswith('PROJ'), tf.trainable_variables())
    W_G = filter(lambda x: x.name.endswith('W:0'), var_G)

    var_E = filter( lambda x: 'ENCODE' in x.name, tf.trainable_variables())
    W_E = filter(lambda x: x.name.endswith('W:0'), var_E)

    if weight_decay_rate > 0:
        loss_G += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_G)))
        loss_adv_D += weight_decay_rate * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_D)))

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth=True

    sess = tf.Session(config=config_proto)

    optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate_proj, beta1=adam_beta1_g_tf, beta2=adam_beta2_g, epsilon=adam_eps_g)
    grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
    grads_vars_G_clipped = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
    train_op_G = optimizer_G.apply_gradients( grads_vars_G_clipped )

    optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate_dis, beta1=adam_beta1_d_tf, beta2=adam_beta2_d, epsilon=adam_eps_d)
    grads_vars_D = optimizer_D.compute_gradients( loss_adv_D, var_list=var_D )
    grads_vars_D_clipped = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
    train_op_D = optimizer_D.apply_gradients( grads_vars_D_clipped )

    D_var_clip_ops = map(lambda v: tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)), W_D)
    E_var_clip_ops = map(lambda v: tf.assign(v, tf.clip_by_value(v, clamp_lower, clamp_upper)), W_E)


    # setup the saver
    saver = tf.train.Saver(max_to_keep=16)
    saver_epoch = tf.train.Saver(max_to_keep=100)

    # setup the image saver
    if output_img > 0:
        num_output_img = min(5, batch_size)
        output_ori_imgs_op = (images_tf[0:num_output_img] * 127.5 ) + 127.5
        output_noisy_imgs_op = (noisy_image_tf[0:num_output_img] * 127.5 ) + 127.5
        output_project_imgs_op = (projection_z[0:num_output_img] * 127.5 ) + 127.5
        output_reconstruct_imgs_op = (projection_x[0:num_output_img] * 127.5 ) + 127.5

    if use_tensorboard > 0:
        # create a summary to monitor cost tensor
        tf.summary.scalar("accuracy_latent", accuracy_latent, collections=['dis'])
        tf.summary.scalar("accuracy_img", accuracy_img, collections=['dis'])
        tf.summary.scalar("loss_adv_D", loss_adv_D, collections=['dis'])
        tf.summary.scalar("loss_adv_D_pos_latent", loss_adv_D_pos_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_latent", loss_adv_D_neg_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_latent", loss_adv_D_latent, collections=['dis'])
        tf.summary.scalar("loss_adv_D_pos_img", loss_adv_D_pos_img, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_img", loss_adv_D_neg_img, collections=['dis'])
        tf.summary.scalar("loss_adv_D_neg_imgz", loss_adv_D_neg_imgz, collections=['dis'])
        tf.summary.scalar("loss_adv_D_img", loss_adv_D_img, collections=['dis'])

        tf.summary.scalar("loss_G", loss_G, collections=['proj'])
        tf.summary.scalar("loss_adv_G", loss_adv_G, collections=['proj'])
        tf.summary.scalar("loss_adv_G_latent", loss_adv_G_latent, collections=['proj'])
        tf.summary.scalar("loss_adv_G_imgx", loss_adv_G_imgx, collections=['proj'])
        tf.summary.scalar("loss_recon_z", loss_recon_z, collections=['proj'])
        tf.summary.scalar("loss_recon", loss_recon, collections=['proj'])
        tf.summary.scalar("loss_proj", loss_proj, collections=['proj'])
        tf.summary.scalar("lambda_ratio", lambda_ratio_tf, collections=['proj'])
        tf.summary.scalar("lambda_l2", lambda_l2_tf, collections=['proj'])
        tf.summary.scalar("lambda_latent", lambda_latent_tf, collections=['proj'])
        tf.summary.scalar("lambda_img", lambda_img_tf, collections=['proj'])
        tf.summary.scalar("lambda_de", lambda_de_tf, collections=['proj'])
        tf.summary.scalar("adam_beta1_g", adam_beta1_g_tf, collections=['proj'])
        tf.summary.scalar("adam_beta1_d", adam_beta1_d_tf, collections=['proj'])
        tf.summary.scalar("learning_rate_proj", learning_rate_proj, collections=['proj'])
        tf.summary.scalar("learning_rate_dis", learning_rate_dis, collections=['proj'])
        tf.summary.image("original_image", images_tf, max_outputs=5, collections=['proj'])
        tf.summary.image("noisy_image", noisy_image_tf, max_outputs=5, collections=['proj'])
        tf.summary.image("projected_z", projection_z, max_outputs=5, collections=['proj'])
        tf.summary.image("reconstructed_x", projection_x, max_outputs=5, collections=['proj'])

        # merge all summaries into a single op
        summary_G = tf.summary.merge_all(key='proj')
        summary_D = tf.summary.merge_all(key='dis')

    # initialization
    sess.run(tf.global_variables_initializer(), feed_dict={
        learning_rate_dis: learning_rate_val_dis,
        adam_beta1_d_tf: adam_beta1_d,
        learning_rate_proj: learning_rate_val_proj,
        lambda_ratio_tf: lambda_ratio,
        lambda_l2_tf: lambda_l2,
        lambda_latent_tf: lambda_latent,
        lambda_img_tf: lambda_img,
        lambda_de_tf: lambda_de,
        adam_beta1_g_tf: adam_beta1_g,
        })
    sess.run(tf.local_variables_initializer(), feed_dict={
        learning_rate_dis: learning_rate_val_dis,
        adam_beta1_d_tf: adam_beta1_d,
        learning_rate_proj: learning_rate_val_proj,
        lambda_ratio_tf: lambda_ratio,
        lambda_l2_tf: lambda_l2,
        lambda_latent_tf: lambda_latent,
        lambda_img_tf: lambda_img,
        lambda_de_tf: lambda_de,
        adam_beta1_g_tf: adam_beta1_g,
        })


    print 'reload previously trained model'
    if use_pretrain == True:
        print 'reloading %s...' % pretrained_model_file
        saver.restore( sess, pretrained_model_file )

    if use_tensorboard > 0:
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        print "Run the command line:\n --> tensorboard --logdir=%s\n" %  logs_base
        print "Then open http://0.0.0.0:6006/ into your web browser"


    # continue the iteration number
    if use_pretrain == True:
        iters = pretrained_iter + 1
    else:
        iters = 0

    start_epoch = iters // (total_train // batch_size)


    print 'start training'
    start_time = timeit.default_timer()

    iters_in_epoch = total_train // batch_size
    epoch = 0

    loss_dis_avg = SmoothStream(window_size=100)
    acc_latent_avg = SmoothStream(window_size=100)
    acc_img_avg = SmoothStream(window_size=100)
    loss_recon_avg = SmoothStream(window_size=100)
    loss_recon_z_avg = SmoothStream(window_size=100)

    update_D_left = Dperiod
    update_G_left = Gperiod

    loss_G_val = 0
    loss_proj_val = 0
    loss_recon_val = 0
    loss_recon_z_val = 0
    loss_adv_G_val = 0
    loss_D_val = 0
    acc_latent_val = 0
    acc_img_val = 0


    print 'alternative training starts....'

    while True:
        inst_batch, inst_noisy_batch = train_queue.get()

        batch = np_combine_batch(inst_batch,ref_batch)
        noisy_batch = np_combine_batch(inst_noisy_batch,ref_batch)

        # adjust learning rate
        learning_rate_val_proj_current = 2e-1 / lambda_de
        learning_rate_val_proj_current = min(learning_rate_val_proj, learning_rate_val_proj_current)


        if update_G_left > 0:

            sys.stdout.write('G: ')

            # update G
            _, loss_G_val, loss_proj_val, loss_recon_val, loss_recon_z_val, loss_adv_G_val, _ = sess.run(
                [train_op_G, loss_G, loss_proj, loss_recon, loss_recon_z, loss_adv_G, updates],
                feed_dict={
                    images_tf: batch,
                    noisy_image_tf: noisy_batch,
                    learning_rate_dis: learning_rate_val_dis,
                    adam_beta1_d_tf: adam_beta1_d,
                    learning_rate_proj: learning_rate_val_proj_current,
                    lambda_ratio_tf: lambda_ratio,
                    lambda_l2_tf: lambda_l2,
                    lambda_latent_tf: lambda_latent,
                    lambda_img_tf: lambda_img,
                    lambda_de_tf: lambda_de,
                    adam_beta1_g_tf: adam_beta1_g,
                })

            update_G_left -= 1
            loss_recon_avg.insert(loss_recon_val)
            loss_recon_z_avg.insert(loss_recon_z_val)


        if update_G_left <= 0 and update_D_left > 0:

            sys.stdout.write('D: ')

            # update D
            _, loss_D_val, acc_latent_val, acc_img_val, _ = sess.run(
                [train_op_D, loss_adv_D, accuracy_latent, accuracy_img, updates],
                feed_dict={
                    images_tf: batch,
                    noisy_image_tf: noisy_batch,
                    learning_rate_dis: learning_rate_val_dis,
                    adam_beta1_d_tf: adam_beta1_d,
                    learning_rate_proj: learning_rate_val_proj_current,
                    lambda_ratio_tf: lambda_ratio,
                    lambda_l2_tf: lambda_l2,
                    lambda_latent_tf: lambda_latent,
                    lambda_img_tf: lambda_img,
                    lambda_de_tf: lambda_de,
                    adam_beta1_g_tf: adam_beta1_g,
                })

            if clamp_weight > 0:
                # clip the variables of the discriminator
                _,_ = sess.run([D_var_clip_ops, E_var_clip_ops])

            update_D_left -= 1

            loss_dis_avg.insert(loss_D_val)
            acc_latent_avg.insert(acc_latent_val)
            acc_img_avg.insert(acc_img_val)

        print "Iter %d (%.2fm): l_gen=%.3e  l_proj=%.3e l_recon=%.3e (%.3e) l_recon_z=%.3e (%.3e) l_adv_gen=%.3e l_dis=%.3e (%.3e) acc_img=%.3e (%.3e) acc_latent=%.3e (%.3e) lrp=%.3e lrd=%.3e qsize=%d" % (
                iters, (timeit.default_timer()-start_time)/60., loss_G_val, loss_proj_val, loss_recon_val,
                loss_recon_avg.get_moving_avg(), loss_recon_z_val, loss_recon_z_avg.get_moving_avg(), loss_adv_G_val, loss_D_val, loss_dis_avg.get_moving_avg(),
                acc_img_val, acc_img_avg.get_moving_avg(),
                acc_latent_val, acc_latent_avg.get_moving_avg(),
                learning_rate_val_proj, learning_rate_val_dis, train_queue.qsize())


        # reset update_D_left and update_G_left when they are zeros
        if update_G_left <= 0 and update_D_left <= 0:
            update_G_left = Gperiod
            update_D_left = Dperiod

        if (iters + 1) % 2000 == 0:
            saver.save(sess, model_path + '/model_iter', global_step=iters)


        # output to tensorboard
        if use_tensorboard >0 and (iters % tensorboard_period == 0):

            summary_d_vals, summary_g_vals = sess.run(
                [summary_D, summary_G],
                feed_dict={
                    images_tf: batch,
                    noisy_image_tf: noisy_batch,
                    learning_rate_dis: learning_rate_val_dis,
                    learning_rate_proj: learning_rate_val_proj_current,
                    lambda_ratio_tf: lambda_ratio,
                    lambda_l2_tf: lambda_l2,
                    lambda_latent_tf: lambda_latent,
                    lambda_img_tf: lambda_img,
                    lambda_de_tf: lambda_de,
                    adam_beta1_g_tf: adam_beta1_g,
                    adam_beta1_d_tf: adam_beta1_d
                })

            summary_writer.add_summary(summary_g_vals, iters)
            summary_writer.add_summary(summary_d_vals, iters)

        # save some images
        if output_img > 0 and (iters + 1) % output_img_period == 0:
            output_ori_img_val, output_noisy_img_val, output_project_img_val, output_reconstruct_imgs_val = sess.run(
                [output_ori_imgs_op, output_noisy_imgs_op, output_project_imgs_op, output_reconstruct_imgs_op],
                feed_dict={
                    images_tf: batch,
                    noisy_image_tf: noisy_batch,
                }
            )
            output_folder = '%s/iter_%d' %(img_path, iters)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for i in range(output_ori_img_val.shape[0]):
                filename = '%s/%d_ori.jpg' % (output_folder, i)
                sp.misc.imsave(filename, output_ori_img_val[i].astype('uint8'))
                filename = '%s/%d_noisy.jpg' % (output_folder, i)
                sp.misc.imsave(filename, output_noisy_img_val[i].astype('uint8'))
                filename = '%s/%d_proj.jpg' % (output_folder, i)
                sp.misc.imsave(filename, output_project_img_val[i].astype('uint8'))
                filename = '%s/%d_recon.jpg' % (output_folder, i)
                sp.misc.imsave(filename, output_reconstruct_imgs_val[i].astype('uint8'))

        iters += 1

        lambda_de *= de_decay_rate


        if iters % iters_in_epoch == 0:
            epoch += 1
            saver_epoch.save(sess, epoch_path + '/model_epoch', global_step=epoch)
            learning_rate_val_dis *= 0.95
            learning_rate_val_proj *= 0.95
            if epoch > n_epochs:
                break

        # recreate new train_proc (force garbage colection)
        if iters % 2000 == 0:
            terminate_train_procs(train_procs)
            del train_procs
            del train_queue
            train_queue = Queue(num_prepare+1)
            train_procs = []
            create_train_procs(trainset, train_queue, n_thread, num_prepare, train_procs)
            
    sess.close()

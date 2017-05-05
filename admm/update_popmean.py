
import sys
sys.path.append("../projector")
import main as model
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.io
import math
import load_celeb
import os
import timeit
import matplotlib.pyplot as plt

import add_noise

import solver_paper as solver
import solver_l1 as solver_l1


import scipy as sp

#np.random.seed(1085)


img_size = (64,64,3)


# filename of the trained model. If using virtual batch normalization, 
use_latent = 1  # if lambda_latent > 0
iter = 49999
pretrained_folder = os.path.expanduser("../projector/model/imsize64_ratio0.010000_dis0.005000_latent0.000100_img0.001000_de1.000000_derate1.000000_dp1_gd1_softpos0.850000_wdcy_0.000000_seed0")
pretrained_model_file = '%s/model/model_iter-%d' % (pretrained_folder, iter)

# the filename of saved the reference batch
ref_file = '%s/ref_batch_25.mat' % pretrained_folder
ref_batch = sp.io.loadmat(ref_file)['ref_batch']
n_reference = ref_batch.shape[0]


# setup the variables in the session
batch_size = n_reference
images_tf = tf.placeholder( tf.float32, [batch_size, img_size[0], img_size[1], img_size[2]], name="images")
is_train = True
proj, latent = model.build_projection_model(images_tf, is_train, n_reference, use_bias=True, reuse=None)
dis, _ = model.build_classifier_model_imagespace(proj, is_train, n_reference, reuse=None)

if use_latent > 0:
    dis_latent,_ = model.build_classifier_model_latentspace(latent, is_train, n_reference, reuse=None)


# We create a session to use the graph and restore the variables
print 'loading model...'
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, pretrained_model_file)
print 'finished reload.'


# updating popmean for faster evaluation

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
updates = tf.group(*update_ops)
with tf.control_dependencies([updates]):
    proj_update = proj * 1
    latent_update = latent * 1
    dis_udpate = dis  * 1
    if use_latent > 0:
        dis_latent_update = dis_latent * 1

if use_latent > 0:
    _,_,_,_,_, = sess.run([proj_update, latent_update, dis_udpate, dis_latent_update, updates],
                      feed_dict={images_tf: ref_batch})
else:
    _,_,_,_, = sess.run([proj_update, latent_update, dis_udpate, updates],
            feed_dict={images_tf: ref_batch})

updated_folder = '%s/update' % (pretrained_folder)
if not os.path.exists(updated_folder):
    os.makedirs(updated_folder)
update_file = '%s/model_iter-%d' % (updated_folder, iter)
saver.save(sess, update_file)

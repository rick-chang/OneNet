import sys
import os
import numpy as np
import glob
import os
import timeit
import scipy as sp
import pickle



dataset_base_path = os.path.expanduser("~/datasets/imagenet")
trainset_path = dataset_base_path + '/' + 'train'
validset_path = dataset_base_path + '/' + 'valid'
testset_path = dataset_base_path + '/' + 'test'

trainset_pickle_path = dataset_base_path + '/' + 'train_filename.pickle'
validset_pickle_path = dataset_base_path + '/' + 'valid_filename.pickle'
testset_pickle_path = dataset_base_path + '/' + 'test_filename.pickle'


def load_pickle(pickle_path, dataset_path):
	if not os.path.exists(pickle_path):

		import magic

		image_files = []
		for dir, _, _, in os.walk(dataset_path):
			filenames = glob.glob( os.path.join(dir, '*.JPEG'))  # may be JPEG, depending on your image files
			image_files.append(filenames)

			## use magic to perform a simple check of the images
			# import magic
			# for filename in filenames:
			#	if magic.from_file(filename, mime=True) == 'image/jpeg':
			#		image_files.append(filename)
			#	else:
			#		print '%s is not a jpeg!' % filename
			#		print magic.from_file(filename)

		if len(image_files) > 0:
			image_files = np.hstack(image_files)

		dataset_filenames = {'image_path':image_files}
		pickle.dump( dataset_filenames, open( pickle_path, "wb" ) )
	else:
		dataset_filenames = pickle.load( open( pickle_path, "rb" ) )
	return dataset_filenames


# return a pd object
def load_trainset_path():
	return load_pickle(trainset_pickle_path, trainset_path)

def load_validset_path():
	return load_pickle(validset_pickle_path, validset_path)

def load_testset_path():
	return load_pickle(testset_pickle_path, testset_path)


# return a list containing all the filenames
def load_trainset_path_list():
	return load_trainset_path()['image_path'].tolist()

def load_validset_path_list():
	return load_validset_path()['image_path'].tolist()

def load_testset_path_list():
	return load_testset_path()['image_path'].tolist()

def load_image( path, pre_height=146, pre_width=146, height=128, width=128 ):

	import skimage.io
	import skimage.transform

	try:
		img = skimage.io.imread( path ).astype( float )
	except:
		return None

	img /= 255.

	if img is None: return None
	if len(img.shape) < 2: return None
	if len(img.shape) == 4: return None
	if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
	if img.shape[2] == 4: img=img[:,:,:3]
	if img.shape[2] > 4: return None

	short_edge = min( img.shape[:2] )
	yy = int((img.shape[0] - short_edge) / 2)
	xx = int((img.shape[1] - short_edge) / 2)
	crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
	resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

	rand_y = np.random.randint(0, pre_height - height)
	rand_x = np.random.randint(0, pre_width - width)

	resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

	resized_img -= 0.5
	resized_img /= 2.0

	return resized_img #(resized_img - 127.5)/127.5


def cache_batch(trainset, queue, batch_size, num_prepare, rseed=None, identifier=None):

	np.random.seed(rseed)

	current_idx = 0
	n_train = len(trainset)
	trainset.index = range(n_train)
	trainset = trainset.ix[np.random.permutation(n_train)]
	idx = 0
	while True:

		# read in data if the queue is too short
		while queue.qsize() < num_prepare:
			start = timeit.default_timer()
			image_paths = trainset[idx:idx+batch_size]['image_path'].values
			images_ori = map(lambda x: load_image( x ), image_paths)
			X = np.asarray(images_ori)
			# put in queue
			queue.put(X) # block until free slot is available
			idx += batch_size
			if idx + batch_size > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
				trainset = trainset.ix[np.random.permutation(n_train)]
				idx = 0



def cache_train_batch_cube(queue, batch_size, num_prepare, identifier=None):
	trainset = load_pickle(trainset_path, dataset_path)
	cache_batch(trainset, queue, batch_size, num_prepare)

def cache_test_batch_cube(queue, batch_size, num_prepare, identifier=None):
	testset = load_pickle(testset_path, dataset_path)
	cache_batch(testset, queue, batch_size, num_prepare)


def cache_batch_list_style(trainset, Xlist, batch_size, num_prepare, identifier=None):

	current_idx = 0
	n_train = len(trainset)
	trainset.index = range(n_train)
	trainset = trainset.ix[np.random.permutation(n_train)]
	idx = 0
	while True:

		# read in data if the queue is too short
		while len(Xlist) < num_prepare:
			image_paths = trainset[idx:idx+batch_size]['image_path'].values
			images_ori = map(lambda x: load_image( x ), image_paths)
			X = np.asarray(images_ori, dtype=float)
			Xlist.append(X)
			idx += batch_size
			if idx + batch_size > n_train: #reset when last batch is smaller than batch_size or reaching the last batch
				trainset = trainset.ix[np.random.permutation(n_train)]
				idx = 0

if __name__ == "__main__":
	trainset = load_trainset_path_list() #
	validset = load_validset_path_list()
	testset = load_testset_path_list()   #

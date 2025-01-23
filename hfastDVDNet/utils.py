"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Pau Céspedes, based on code by Matias Tassano 

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import subprocess
import glob
import logging
import numpy as np
import cv2
import torch
from skimage.measure.simple_metrics import compare_psnr
from tensorboardX import SummaryWriter
import tifffile
from torch import nn

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def normalize_data(datain_o, datain_n, datain_d, ctrl_fr_idx):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	img_train_o = datain_o
	img_train_d = datain_d
	img_train_n = datain_n

	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
	img_train_o = img_train_o.view(img_train_o.size()[0], -1, img_train_o.size()[-2], img_train_o.size()[-1]) / 255.
	img_train_d = img_train_d.view(img_train_d.size()[0], -1, img_train_d.size()[-2], img_train_d.size()[-1]) / 255.
	img_train_n = img_train_n.view(img_train_n.size()[0], -1, img_train_n.size()[-2], img_train_n.size()[-1]) / 255.

	# extract ground truth (central frame)
	gt_train_o = img_train_o[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
	gt_train_n = img_train_n[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
	gt_train_d = img_train_d[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
	return img_train_o, img_train_n, img_train_d, gt_train_o, gt_train_n, gt_train_d

def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:max_num_fr]:

		img, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Load image
		if fpath.endswith('.tif'):
			img = tifffile.imread(fpath)
		else:
			img = cv2.imread(fpath)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.transpose(2, 0, 1)

	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	return img, expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict

def get_noise_level(path):
	# Form the path to the subfolder's text file
	txt_file_path = os.path.join(path, "noise_level.txt")
	# Check if the text file exists
	if os.path.isfile(txt_file_path):
		# Read the integer from the text file and store it in the dictionary
		with open(txt_file_path, 'r') as file:
			content = file.read()
			return int(content.strip())/255.

def deactivate_batchnorm_saved_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
    return model

def remove_batchnorm_layers(model):
    """
    Replaces all nn.BatchNorm2d layers in the model with nn.Identity() layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            parent_name = name.rsplit('.', 1)[0]
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
            else:
                parent_module = model

            setattr(parent_module, name.split('.')[-1], nn.Identity())

    return model

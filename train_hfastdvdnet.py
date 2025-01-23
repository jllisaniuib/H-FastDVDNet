"""
Trains a H-FastDVDnet model.

author: Pau Céspedes, based on code from Matias Tassano

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from hfastDVDNet.models import HFastDVDnet
from hfastDVDNet.dataset import ValDataset
from hfastDVDNet.dataloaders import train_dali_loader
from hfastDVDNet.images_dataloader import ImagesDataLoader
from hfastDVDNet.utils import svd_orthogonalization, close_logger, init_logging, normalize_data
from hfastDVDNet.train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint, log_training_patches
from PIL import Image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gc
from datetime import datetime

def main(**args):
	r"""Performs the main training loop
	"""

	# Load dataset
	print('> Loading datasets ...')
	dataset_val = ValDataset(valsetdir_noisy=args['valset_dir_noisy'], \
							 valsetdir_denoised=args['valset_dir_denoised'], \
							 valsetdir_original=args['valset_dir_original'], \
							 gray_mode=False)

	images_loader_train = ImagesDataLoader(batch_size=args['batch_size'], \
										   sequence_length=args['temp_patch_size'], \
										   clean_files=args['trainset_dir_original'], \
										   noisy_files=args['trainset_dir_noisy'], \
										   denoised_files=args['trainset_dir_denoised'], \
										   crop_size=args['patch_size'])

	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [0]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet()
	model = nn.DataParallel(model, device_ids=device_ids).cuda()
	print_model_parameters(model)

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train
		# start = time.time()
		for i, data in enumerate(images_loader_train, 0):
			# if i % 100 == 0:
			# 	end = time.time()
			# 	print("Elapsed time till %d steps: %f s" % (i, end - start))

			# Manually stop when epoch is completed
			if i >= num_minibatches:
				break

			# Uncomment to quickly test testing phase
			# if i > 1:
			# 	validate_and_log(
			# 		model_temp=model, \
			# 		dataset_val=dataset_val, \
			# 		temp_psz=args['temp_patch_size'], \
			# 		writer=writer, \
			# 		epoch=epoch, \
			# 		lr=current_lr, \
			# 		logger=logger, \
			# 		trainimg=imgo_train
			# 	)

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			imgo_train, imgn_train, imgd_train, gt_train, gt_n, gt_d = normalize_data(data[0], data[1], data[2], ctrl_fr_idx)

			N, _, H, W = imgn_train.size()

			# Test for different frames
			# do = imgo_train[0]
			# showImage(do[0:3, :, :], "1. Original from pipeline: ")
			# showImage(do[3:6, :, :], "2. Original from pipeline: ")
			# showImage(do[6:9, :, :], "3. Original from pipeline: ")
			# showImage(do[9:12, :, :], "4. Original from pipeline: ")
			# showImage(do[12:15, :, :], "5. Original from pipeline: ")

			# Test for different categories(original, noisy, denoised)
			# imgn = data[1][0][2]
			# showImage(imgn, "PRE-N")
			# showImage(gt_train[0], "GT0")
			# showImage(gt_n[0], "GTN")
			# showImage(gt_d[0], "GTD")

			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)
			imgd_train = imgd_train.cuda(non_blocking=True)
			noise_map = data[3].expand((N, 1, H, W)).cuda(non_blocking=True)  # one channel per image

			# Evaluate model and optimize it
			out_train = model(imgn_train, imgd_train, noise_map)

			# Compute loss
			loss = criterion(gt_train, out_train) / (N*2)
			loss.backward()
			optimizer.step()

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)

			if training_params['step'] % 1000 == 0:
				log_training_patches(writer, epoch + 1, training_params['step'], imgo_train, imgn_train, imgd_train)

			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Validation and log images
		validate_and_log(
						model_temp=model, \
						dataset_val=dataset_val, \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger, \
						trainimg=imgo_train
						)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)


	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

def showImage(img, t):
	image = img.cpu().numpy()  # Extract the image at index i

	# Assuming the tensor is in [0, 1] range, you can convert it to [0, 255] range
	image = (image * 255)
	image = image.astype(np.int).transpose(1, 2, 0)

	# Display the image
	plt.imshow(image)
	plt.title(f"Image {t}")
	plt.show()

# def areImagesDesincronized(img1, img2, step, epoch):
# 	image1 = img1.cpu()  # Extract the image at index i
# 	image1 = (image1 * 255).byte()
# 	image1 = image1.permute(1, 2, 0).numpy()
# 	i1 = Image.fromarray(image1)
#
# 	image2 = img2.cpu()  # Extract the image at index i
# 	image2 = (image2 * 255).byte()
# 	image2 = image2.permute(1, 2, 0).numpy()
# 	i2 = Image.fromarray(image2)
#
# 	hash0 = imghash.average_hash(i1)
# 	hash1 = imghash.average_hash(i2)
#
# 	cutoff = 30  # maximum bits that could be different between the hashes.
# 	diff = abs(hash0 - hash1)
#
# 	if diff >= cutoff:
# 		print('Iajuuuuu')
# 		f = open("/home/pau/TFG/logs/NEWLOGS/desync-finder/log/desync-info.txt", "a")
# 		f.write("-Desync at step {0} of epoch {1} (hash diff {2})\n".format(step, epoch, diff))
# 		f.close()
#
# 		current_datetime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
# 		i1.save('/home/pau/TFG/logs/NEWLOGS/desync-finder/images/original/{0}.jpg'.format(current_datetime), 'JPEG')
# 		i2.save('/home/pau/TFG/logs/NEWLOGS/desync-finder/images/denoised/{0}.jpg'.format(current_datetime), 'JPEG')
# 		return True
# 	return False

def print_model_parameters(model):
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print('Total number of parameters: {}'.format(pytorch_total_params))

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print('Total number of Trainable parameters: {}'.format(pytorch_total_params))

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=80, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=256000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir_noisy", type=str, default=None, \
					 help='path of noisy trainset')
	parser.add_argument("--trainset_dir_denoised", type=str, default=None, \
						help='path of denoised trainset')
	parser.add_argument("--trainset_dir_original", type=str, default=None, \
						help='path of original trainset')
	parser.add_argument("--valset_dir_noisy", type=str, default=None, \
						 help='path of validation set')
	parser.add_argument("--valset_dir_denoised", type=str, default=None, \
						help='path of validation set')
	parser.add_argument("--valset_dir_original", type=str, default=None, \
						help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))

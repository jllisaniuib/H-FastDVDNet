"""
H-FastDVDnet denoising algorithm

Pau Céspedes, based on code by Matias Tassano 
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def temp_denoise(model, noisyframe, denoisedframe, sigma_noise):
	'''Encapsulates call to denoising model and handles padding.
		Expects noisyframe to be normalized in [0., 1.]
	'''
	# make size a multiple of four (we have two scales in the denoiser)
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%4
	if expanded_h:
		expanded_h = 4-expanded_h
	expanded_w = sh_im[-1]%4
	if expanded_w:
		expanded_w = 4-expanded_w
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	denoisedframe = F.pad(input=denoisedframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = torch.clamp(model(noisyframe, denoisedframe, sigma_noise), 0., 1.)

	if expanded_h:
		out = out[:, :, :-expanded_h, :]
	if expanded_w:
		out = out[:, :, :, :-expanded_w]

	return out

def denoise_seq_hfastdvdnet(noisyseq, denoisedseq, noise_std, temp_psz, model_temporal):
	r"""Denoises a sequence of frames with H-FastDVDnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
	Returns:
		denframes: Tensor, [numframes, C, H, W]
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, C, H, W = noisyseq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes_noisy = list()
	inframes_denoised = list()
	denframes = torch.empty((numframes, C, H, W)).to(noisyseq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, 1, H, W))

	for fridx in range(numframes):
		# load input frames
		if not inframes_noisy:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = abs(idx-ctrlfr_idx) # handle border conditions, reflect
				inframes_noisy.append(noisyseq[relidx])
				inframes_denoised.append(denoisedseq[relidx])
		else:
			del inframes_noisy[0]
			del inframes_denoised[0]
			relidx = min(fridx + ctrlfr_idx, -fridx + 2*(numframes-1)-ctrlfr_idx) # handle border conditions
			inframes_noisy.append(noisyseq[relidx])
			inframes_denoised.append(denoisedseq[relidx])

		inframes_t_noisy = torch.stack(inframes_noisy, dim=0).contiguous().view((1, temp_psz*C, H, W)).to(noisyseq.device)
		inframes_t_denoised = torch.stack(inframes_denoised, dim=0).contiguous().view((1, temp_psz * C, H, W)).to(denoisedseq.device)

		# append result to output list
		denframes[fridx] = temp_denoise(model_temporal, inframes_t_noisy, inframes_t_denoised, noise_map)

	# free memory up
	del inframes_noisy
	del inframes_t_noisy
	del inframes_denoised
	del inframes_t_denoised
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes

# CODI VISUALITZAR IMATGE
def visualizeImageTensor(tensor):
	# Assuming tensor is your input tensor with shape (1, 15, 480, 912) and values between 0 and 1
	# Convert the tensor to a NumPy array
	tensor_np = tensor.squeeze().cpu().numpy()  # Squeeze removes the singleton dimension (1)

	# Create an image from the array by scaling the values back to 0-255 range
	image_array = (tensor_np * 255).astype(np.uint8)

	# Create a 3-channel image from the single-channel array
	image = cv2.merge([image_array] * 3)

	# Display the image
	cv2.imshow('Image', image)
	cv2.waitKey(0)  # Wait until a key is pressed
	cv2.destroyAllWindows()  # Close the window

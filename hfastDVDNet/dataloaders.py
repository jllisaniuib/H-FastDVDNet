'''Implements a sequence dataloader using NVIDIA's DALI library.

Pau Céspedes, based on code by Matias Tassano 

The dataloader is based on the VideoReader DALI's module, which is a 'GPU' operator that loads
and decodes H264 video codec with FFmpeg.

Based on
https://github.com/NVIDIA/DALI/blob/master/docs/examples/video/superres_pytorch/dataloading/dataloaders.py
'''
import os
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class VideoReaderPipeline(Pipeline):
	''' Pipeline for reading H264 videos based on NVIDIA DALI.
	Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W]
	(N being the batch size and F the number of frames). Frames are RGB uint8.
	Args:
		batch_size: (int)
				Size of the batches
		sequence_length: (int)
				Frames to load per sequence.
		num_threads: (int)
				Number of threads.
		device_id: (int)
				GPU device ID where to load the sequences.
		files_noisy: (str or list of str)
				File names of the noisy video files to load.
		files_denoised: (str or list of str)
				File names of the denoised video files to load.
		files_original: (str or list of str)
				File names of the original video files to load.
		crop_size: (int)
				Size of the crops. The crops are in the same location in all frames in the sequence
		random_shuffle: (bool, optional, default=True)
				Whether to randomly shuffle data.
		step: (int, optional, default=-1)
				Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
	'''
	def __init__(self, batch_size, sequence_length, num_threads, device_id, files_noisy,
				 files_denoised, files_original, crop_size, random_shuffle=True, step=-1):
		super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=2323)
		# Define VideoReader for noisy videos
		self.reader_noisy = ops.VideoReader(device="gpu",
											filenames=files_noisy,
											sequence_length=sequence_length,
											normalized=False,
											random_shuffle=random_shuffle,
											seed=self.seed,
											image_type=types.DALIImageType.RGB,
											dtype=types.DALIDataType.UINT8,
											step=step,
											initial_fill=16,
											pad_last_batch=True)

		# Define VideoReader for denoised videos
		self.reader_denoised = ops.VideoReader(device="gpu",
											filenames=files_denoised,
											sequence_length=sequence_length,
											normalized=False,
											random_shuffle=random_shuffle,
											seed=self.seed,
											image_type=types.DALIImageType.RGB,
											dtype=types.DALIDataType.UINT8,
											step=step,
											initial_fill=16,
											pad_last_batch=True)

		# Define VideoReader for denoised videos
		self.reader_original = ops.VideoReader(device="gpu",
											   filenames=files_original,
											   sequence_length=sequence_length,
											   normalized=False,
											   random_shuffle=random_shuffle,
											   seed=self.seed,
											   image_type=types.DALIImageType.RGB,
											   dtype=types.DALIDataType.UINT8,
											   step=step,
											   initial_fill=16,
											   pad_last_batch=True)

		# Define crop and permute operations to apply to every sequence
		self.crop = ops.CropMirrorNormalize(device="gpu",
											crop_w=crop_size,
											crop_h=crop_size,
											output_layout='FCHW',
											dtype=types.DALIDataType.FLOAT)
		self.uniform = ops.Uniform(range=(0.0, 1.0))  # used for random crop

	def define_graph(self):
		'''Definition of the graph--events that will take place at every sampling of the dataloader.
		The random crop and permute operations will be applied to the sampled sequence.
		'''
		x_pos = self.uniform()
		y_pos = self.uniform()
		input_noisy = self.reader_noisy(name="ReaderNoisy")
		input_denoised = self.reader_denoised(name="ReaderDenoised")
		input_original = self.reader_original(name="ReaderOriginal")
		cropped_noisy = self.crop(input_noisy, crop_pos_x=x_pos, crop_pos_y=y_pos)
		cropped_denoised = self.crop(input_denoised, crop_pos_x=x_pos, crop_pos_y=y_pos)
		cropped_original = self.crop(input_original, crop_pos_x=x_pos, crop_pos_y=y_pos)
		return cropped_noisy, cropped_denoised, cropped_original

class train_dali_loader():
	'''Sequence dataloader.
	Args:
		batch_size: (int)
			Size of the batches
		file_root: (str)
			Path to directory with video sequences
		sequence_length: (int)
			Frames to load per sequence
		crop_size: (int)
			Size of the crops. The crops are in the same location in all frames in the sequence
		epoch_size: (int, optional, default=-1)
			Size of the epoch. If epoch_size <= 0, epoch_size will default to the size of VideoReaderPipeline
		random_shuffle (bool, optional, default=True)
			Whether to randomly shuffle data.
		temp_stride: (int, optional, default=-1)
			Frame interval between each sequence
			(if `temp_stride` < 0, `temp_stride` is set to `sequence_length`).
	'''

	def __init__(self, batch_size, noisy_file_root, denoised_file_root, original_file_root, sequence_length,
				 crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
		# Builds list of sequence filenames
		container_noisy_files = os.listdir(noisy_file_root)
		container_noisy_files = [noisy_file_root + '/' + f for f in container_noisy_files]

		container_denoised_files = os.listdir(denoised_file_root)
		container_denoised_files = [denoised_file_root + '/' + f for f in container_denoised_files]

		container_original_files = os.listdir(original_file_root)
		container_original_files = [original_file_root + '/' + f for f in container_original_files]

		# Define and build pipeline
		self.pipeline = VideoReaderPipeline(batch_size=batch_size,
											sequence_length=sequence_length,
											num_threads=2,
											device_id=0,
											files_noisy=container_noisy_files,
											files_denoised=container_denoised_files,
											files_original=container_original_files,
											crop_size=crop_size,
											step=temp_stride,
											random_shuffle=random_shuffle)
		self.pipeline.build()

		# Define size of epoch
		if epoch_size <= 0:
			self.epoch_size = self.pipeline.epoch_size("ReaderNoisy")
		else:
			self.epoch_size = epoch_size
		self.dali_iterator = pytorch.DALIGenericIterator(pipelines=self.pipeline,
														 output_map=["data_noisy", "data_denoised" , "data_original"],
														 size=self.epoch_size,
														 auto_reset=True)

	def __len__(self):
		return self.epoch_size
	def __iter__(self):
		return self.dali_iterator.__iter__()

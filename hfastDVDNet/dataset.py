"""
Dataset related functions

Pau Céspedes, based on code by Matias Tassano 

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from utils import open_sequence, get_noise_level

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir_noisy=None, valsetdir_denoised=None, valsetdir_original=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		self.noisy_sequences = self.loadSequences(valsetdir_noisy, gray_mode, num_input_frames)
		self.noise_levels = self.get_noises(valsetdir_noisy)
		self.denoised_sequences = self.loadSequences(valsetdir_denoised, gray_mode, num_input_frames)
		self.original_sequences = self.loadSequences(valsetdir_original, gray_mode, num_input_frames)

	def __getitem__(self, index):
		noisy = torch.from_numpy(self.noisy_sequences[index])
		denoised = torch.from_numpy(self.denoised_sequences[index])
		original = torch.from_numpy(self.original_sequences[index])
		noise = self.noise_levels[index]

		return noisy, denoised, original, noise

	def __len__(self):
		return len(self.original_sequences)

	def loadSequences(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		noises = {}
		sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=num_input_frames)
			noises[os.path.basename(seq_dir)] = get_noise_level(seq_dir)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)

		return sequences

	def get_noises(self, valsetdir):
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		noises = []
		for seq_dir in seqs_dirs:
			noises.append(get_noise_level(seq_dir))

		return noises

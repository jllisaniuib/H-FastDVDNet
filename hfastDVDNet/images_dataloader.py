"""
Dataset read/write related functions

Pau Céspedes, based on code by Matias Tassano 

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import random
import tifffile
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import numpy as np

IMAGE_EXTENSIONS = ('.png', '.tif')

class ImagesDataLoader:
    def __init__(self, batch_size, sequence_length, clean_files, noisy_files, denoised_files, crop_size):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.denoised_files = denoised_files
        self.crop_size = crop_size
        self.video_names = [d for d in os.listdir(clean_files) if
                            os.path.isdir(os.path.join(clean_files, d)) and
                            os.path.isdir(os.path.join(noisy_files, d)) and
                            os.path.isdir(os.path.join(denoised_files, d))]
        self.noise_levels = self.create_noises_dictionary(noisy_files)
        # print(self.noise_levels)
        self.transform = A.ReplayCompose(
            [
                A.RandomCrop(height=crop_size, width=crop_size),
                A.OneOf([
                    A.HorizontalFlip(p=0.2),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.6),
                ], p=0.83333),
                ToTensorV2()
            ],
            additional_targets={'noisy': 'image', 'denoised': 'image'}
        )

    def __iter__(self):
        return self

    def __next__(self):
        # Randomly select batch_size folders from either clean_files or noisy_files
        batch_randomly_selected_videos = random.sample(self.video_names, self.batch_size)

        # Initialize batches as PyTorch tensors
        clean_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noisy_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        denoised_batch = torch.zeros((self.batch_size, self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noise_levels = torch.zeros((self.batch_size, 1, 1, 1), device='cuda', dtype=torch.float)

        for i, video_name in enumerate(batch_randomly_selected_videos):
            noise_levels[i] = self.noise_levels[video_name]
            clean_batch[i], noisy_batch[i], denoised_batch[i] = self.get_frames(video_name)

        return clean_batch, noisy_batch, denoised_batch, noise_levels

    def initial_process_image(self, image_path):
        # Load image
        if image_path.endswith('.tif'):
            image = tifffile.imread(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def replay_process_and_augment_images(self, img_o, img_n, img_d, replay):
        img_np_o = self.initial_process_image(img_o)
        img_np_n = self.initial_process_image(img_n)
        img_np_d = self.initial_process_image(img_d)

        augmented_images = A.ReplayCompose.replay(replay, image=img_np_o, noisy=img_np_n, denoised=img_np_d)

        # self.visualize(augmented_images)

        # Convert the numpy arrays to PyTorch tensors
        img_tensor_o = augmented_images['image'].to(device='cuda').float()
        img_tensor_n = augmented_images['noisy'].to(device='cuda').float()
        img_tensor_d = augmented_images['denoised'].to(device='cuda').float()

        return img_tensor_o, img_tensor_n, img_tensor_d

    def first_process_and_augment_images(self, img_o, img_n, img_d):
        img_np_o = self.initial_process_image(img_o)
        img_np_n = self.initial_process_image(img_n)
        img_np_d = self.initial_process_image(img_d)

        augmented_images = self.transform(image=img_np_o, noisy=img_np_n, denoised=img_np_d)

        # self.visualize(augmented_images)

        # Convert the numpy arrays to PyTorch tensors
        img_tensor_o = augmented_images['image'].to(device='cuda').float()
        img_tensor_n = augmented_images['noisy'].to(device='cuda').float()
        img_tensor_d = augmented_images['denoised'].to(device='cuda').float()

        return img_tensor_o, img_tensor_n, img_tensor_d, augmented_images['replay']

    def visualize(self, o, n, d):
        # create figure
        fig = plt.figure(figsize=(10, 5))
        # fig.title('Augmentations: ')

        # Adds a subplot at the 1st position
        fig.add_subplot(1, 3, 1)

        # showing image
        plt.imshow(o[0])
        plt.axis('off')
        plt.title("Original Image")

        # Adds a subplot at the 1st position
        fig.add_subplot(1, 3, 2)

        # showing image
        plt.imshow(n[0])
        plt.axis('off')
        plt.title("Noisy Image")

        # Adds a subplot at the 1st position
        fig.add_subplot(1, 3, 3)

        # showing image
        plt.imshow(d[0])
        plt.axis('off')
        plt.title("Denoised Image")

        plt.show()

    def get_frames(self, video_name):
        clean_path = os.path.join(self.clean_files, video_name)
        noisy_path = os.path.join(self.noisy_files, video_name)
        denoised_path = os.path.join(self.denoised_files, video_name)

        # Get all PNG image files in the folder
        clean_frame_paths = [os.path.join(clean_path, f) for f in os.listdir(clean_path) if f.endswith(IMAGE_EXTENSIONS)]
        clean_frame_paths.sort()
        noisy_frame_paths = [os.path.join(noisy_path, f) for f in os.listdir(noisy_path) if f.endswith(IMAGE_EXTENSIONS)]
        noisy_frame_paths.sort()
        denoised_frame_paths = [os.path.join(denoised_path, f) for f in os.listdir(denoised_path) if f.endswith(IMAGE_EXTENSIONS)]
        denoised_frame_paths.sort()

        ctrl_frame_idx = random.randint(self.sequence_length // 2,
                                        len(clean_frame_paths) - (1 + self.sequence_length // 2))

        clean_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        noisy_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)
        denoised_frames = torch.zeros((self.sequence_length, 3, self.crop_size, self.crop_size), device='cuda', dtype=torch.float)

        first_frame_idx = ctrl_frame_idx - self.sequence_length // 2
        last_frame_idx = ctrl_frame_idx + self.sequence_length // 2
        if self.sequence_length % 2 == 1:
            last_frame_idx += 1

        clean_frames[0], noisy_frames[0], denoised_frames[0], replay = self.first_process_and_augment_images(
            clean_frame_paths[first_frame_idx], noisy_frame_paths[first_frame_idx], denoised_frame_paths[first_frame_idx])

        for i in range(first_frame_idx + 1, last_frame_idx):
            clean_frames[i - first_frame_idx], noisy_frames[i - first_frame_idx], denoised_frames[i - first_frame_idx] = self.replay_process_and_augment_images(
                clean_frame_paths[i], noisy_frame_paths[i], denoised_frame_paths[i], replay)

        return clean_frames, noisy_frames, denoised_frames

    def create_noises_dictionary(self, path):
        result_dict = {}
        # Iterate over all items in the directory
        for item in os.listdir(path):
            # Check if the item is a directory
            if os.path.isdir(os.path.join(path, item)):
                # Form the path to the subfolder's text file
                txt_file_path = os.path.join(path, item, "noise_level.txt")
                # Check if the text file exists
                if os.path.isfile(txt_file_path):
                    # Read the integer from the text file and store it in the dictionary
                    with open(txt_file_path, 'r') as file:
                        content = file.read()
                        try:
                            value = int(content.strip())/255.
                        except ValueError:
                            print(f"Warning: Unable to convert content of {txt_file_path} to an integer.")
                            continue
                        result_dict[item] = value
                else:
                    print(f"Warning: {item}.txt not found in {os.path.join(path, item)}")
        return result_dict

    def visualizeImg(self, img, title):
        plt.imshow(img)
        plt.title(title)
        plt.show()

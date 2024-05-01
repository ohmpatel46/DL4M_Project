import os
import random
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class MixedAudioDataset(Dataset):
    def __init__(self, mixed_dir, clean_dir):
        self.mixed_files = os.listdir(mixed_dir)
        self.clean_dir = clean_dir
        self.mixed_dir=mixed_dir

    def __len__(self):
        return len(self.mixed_files)

    def __getitem__(self, idx):
        mixed_file = self.mixed_files[idx]
        # Load mixed audio
        mixed_audio, sr = sf.read(os.path.join(self.mixed_dir, mixed_file))
        mixed_audio_tensor = torch.FloatTensor(mixed_audio)

        # Extract speaker IDs and chapter IDs from file name
        clean_file1,clean_file2 = mixed_file.split('_')
        speaker_chapter_1 = re.split(r'[-.]', clean_file1)
        speaker_chapter_2 = re.split(r'[-.]', clean_file2)
        clean_dir1=os.path.join(self.clean_dir,speaker_chapter_1[0],speaker_chapter_1[1])
        clean_dir2=os.path.join(self.clean_dir,speaker_chapter_2[0],speaker_chapter_2[1])
        clean_audio1, _ = sf.read(os.path.join(clean_dir1, clean_file1))
        clean_audio2, _ = sf.read(os.path.join(clean_dir2, clean_file2))
        clean_audio1_tensor = torch.FloatTensor(clean_audio1)
        clean_audio2_tensor = torch.FloatTensor(clean_audio2)
        min_length = len(mixed_audio_tensor)
        clean_audio1_tensor=clean_audio1_tensor[:min_length]
        clean_audio2_tensor=clean_audio2_tensor[:min_length]
        return mixed_audio_tensor, clean_audio1_tensor, clean_audio2_tensor
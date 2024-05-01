import os
import random
import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return waveform, sample_rate

# Function to merge two audio clips of equal length
def merge_audio(audio1, audio2):
    min_length = min(len(audio1), len(audio2))
    merged_audio = (audio1[:min_length] + audio2[:min_length]) / 2
    return merged_audio

#Function to generate mixed audio signals
def generate_data(output_dir, librispeech_dir):
    speaker_dirs = [os.path.join(librispeech_dir, speaker) for speaker in os.listdir(librispeech_dir) if os.path.isdir(os.path.join(librispeech_dir, speaker))]
    for i, speaker_dir1 in enumerate(speaker_dirs):
        if i==0 or i==1:
            continue
        for speaker_dir2 in speaker_dirs[i+1:]:
            # Get list of chapter directories for each speaker
            chapter_dirs1 = [os.path.join(speaker_dir1, chapter) for chapter in os.listdir(speaker_dir1) if os.path.isdir(os.path.join(speaker_dir1, chapter))]
            chapter_dirs2 = [os.path.join(speaker_dir2, chapter) for chapter in os.listdir(speaker_dir2) if os.path.isdir(os.path.join(speaker_dir2, chapter))]
            for j in range(0,9):
                # Randomly select a chapter from each speaker
                chapter_dir1 = random.choice(chapter_dirs1)
                chapter_dir2 = random.choice(chapter_dirs2)

                # Get list of audio files in each chapter directory
                audio_files1 = [os.path.join(chapter_dir1, audio) for audio in os.listdir(chapter_dir1) if audio.endswith('.flac')]
                audio_files2 = [os.path.join(chapter_dir2, audio) for audio in os.listdir(chapter_dir2) if audio.endswith('.flac')]

                # Randomly select an audio file from each chapter
                audio_file1 = random.choice(audio_files1)
                audio_file2 = random.choice(audio_files2)

                # Load the audio files
                audio1, sr1 = load_audio(audio_file1)
                audio2, sr2 = load_audio(audio_file2)

                # Merge the audio clips
                merged_audio = merge_audio(audio1, audio2)

                # Generate output file name
                output_file = f"{os.path.basename(audio_file1)}_{os.path.basename(audio_file2)}"

                # Save the mixed audio file
                output_path = os.path.join(output_dir, output_file)
                # librosa.output.write_wav(output_path, merged_audio, sr1)
                sf.write(output_path, merged_audio, sr1)
            if i>2:
                break
        if i>2:
            break
    # Returns number of files generated
    return len(os.listdir(output_dir))

def collate_fn(batch):
        # Unzip the batch into separate lists of mixed_audio, clean_audio1, and clean_audio2
        mixed_audios, clean_audios1, clean_audios2 = zip(*batch)
        
        # Determine the maximum length among all audio signals in the batch
        max_length = max(max(len(m), len(c1), len(c2)) for m, c1, c2 in zip(mixed_audios, clean_audios1, clean_audios2))
        
        # Pad all audio signals in the batch to the maximum length
        mixed_audios_padded = [torch.nn.functional.pad(m, (0, max_length - len(m))) for m in mixed_audios]
        clean_audios1_padded = [torch.nn.functional.pad(c1, (0, max_length - len(c1))) for c1 in clean_audios1]
        clean_audios2_padded = [torch.nn.functional.pad(c2, (0, max_length - len(c2))) for c2 in clean_audios2]
        
        # Stack the padded tensors to create the batch
        mixed_audios_batch = torch.stack(mixed_audios_padded)
        clean_audios1_batch = torch.stack(clean_audios1_padded)
        clean_audios2_batch = torch.stack(clean_audios2_padded)
        
        return mixed_audios_batch, clean_audios1_batch, clean_audios2_batch


def si_sdr_loss(y_pred, y_true):
    """
    Compute Scale-Invariant Source-to-Distortion Ratio (SI-SDR) loss.
    Args:
        y_pred (torch.Tensor): Predicted separated audio sources (batch_size, num_sources, num_samples).
        y_true (torch.Tensor): True separated audio sources (batch_size, num_sources, num_samples).
    Returns:
        torch.Tensor: Mean SI-SDR loss over the batch.
    """
    eps = 1e-10
    # Target projection
    s_target = torch.sum(y_pred * y_true, dim=-1, keepdim=True) * y_true / torch.sum(y_true ** 2, dim=-1, keepdim=True)
    # Error signal
    e_noise = y_pred - s_target
    # SI-SDR numerator
    numerator = torch.sum(s_target ** 2, dim=-1)
    # SI-SDR denominator
    denominator = torch.sum(e_noise ** 2, dim=-1)
    # Clipping to avoid numerical instability
    numerator = torch.clamp(numerator, min=eps)
    denominator = torch.clamp(denominator, min=eps)
    # SI-SDR
    si_sdr = 10 * torch.log10(numerator / denominator)
    # Negative SI-SDR as loss
    loss = -torch.mean(si_sdr)
    return loss
# Speech Audio Source Seperation

DL4M Group 6 - Ohm Patel, Emily Wang, Yilin Wang

## Introduction
Our project is based on the cocktail party problem. Based on real-life scenarios such as multiple people talking simultaneously at a party, this problem is often used to train and evaluate audio source separation models. Our models have a variety of applications such as separating political speakers speaking over one another and distinguishing a user’s voice from other people.

## Dataset

- [LibriSpeech Corpus](https://www.openslr.org/12) - The LibriSpeech corpus is a collection of approximately 1,000 hours of audiobooks that are a part of the LibriVox project. Most of the audiobooks come from the Project Gutenberg. The training data is split into 3 partitions of 100hr, 360hr, and 500hr sets while the dev and test data are split into the ’clean’ and ’other’ categories, respectively, depending upon how well or challenging Automatic Speech Recognition systems would perform against. Each of the dev and test sets is around 5hr in audio length. This corpus also provides the n-gram language models and the corresponding texts excerpted from the Project Gutenberg books, which contain 803M tokens and 977K unique words.
- [LibriMix Dataset](https://github.com/JorisCos/LibriMix) - LibriMix is an open-source alternative to wsj0-2mix. Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!.

## Reference Papers

- [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://doi.org/10.48550/arXiv.2005.11262) - In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it. However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!. Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!. Using Conv-TasNet, we achieve competitive performance on all LibriMix versions. In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise. Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic, conversation-like scenarios, we also release a sparsely overlapping version of LibriMix's test set.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.1505.04597) - There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU.

## Environment Setup
Before setting up the environment, download the train-clean-100 folder from the above mentioned link and place in the ./Data directory inside the project's main directory.

To set up the environment for this project, follow these steps:

1. Clone this repository:
```bash
git clone https://github.com/ohmpatel46/DL4M_Project.git
```

2. Navigate to the project directory:
```bash
cd ./DL4M_Project
```

3. Create a new Conda environment using the environment.yml file:
```bash
conda env create -f environment.yml
```
This will create a new environment with all the required dependencies.


4. Activate the newly created environment:
```bash
conda activate myenv
```
Replace `myenv` with the name of your environment.


5. Start working on the project!

## Model
We plan to make architectural changes to an existing model called ConvTasNet. It uses a linear encoder to generate a representation of the speech waveform optimized for separating individual speakers. 
Audio Source Separation: The model provides us with 2 source separated audio clips.
Transcript Generation: We use a pre-trained model on the separated audio clips to generate their transcripts.
Evaluation: The performance metric that is most appropriate for this task is SI-SDR (Scale invariant- Source to Distortion ratio). It measures the ratio between the power of the source signal to the distortion introduced by the source separation.

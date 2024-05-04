# Speech Audio Source Seperation

DL4M Group 6 - Ohm Patel, Emily Wang, Yilin Wang

## Introduction
Our project is based on the cocktail party problem. Based on real-life scenarios such as multiple people talking simultaneously at a party, this problem is often used to train and evaluate audio source separation models. Our models have a variety of applications such as separating political speakers speaking over one another and distinguishing a user’s voice from other people.

## Dataset

- [LibriSpeech Corpus](https://www.openslr.org/12) - Description of dataset 1.
- [LibriMix Dataset](https://github.com/JorisCos/LibriMix) - Description of dataset 2.

## Reference Papers

- [LibriMix: An Open-Source Dataset for Generalizable Speech Separation](https://doi.org/10.48550/arXiv.2005.11262) - Description of paper 1.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.48550/arXiv.1505.04597) - Description of paper 2.

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

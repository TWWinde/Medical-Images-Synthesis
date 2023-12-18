# Unsupervised Semantic Image Synthesis for Medical Imaging


Obtaining large labeled datasets in the medical field is often hard due to privacy concerns. A 
promising solution is to generate synthetic labeled data with Generative Adversarial Networks. 
Given a labeled dataset A containing images from one modality (e.g. CT scans) and their semantic 
labels, and an unlabeled dataset B with unlabeled images from another modality (MRI scans), the 
task is to translate the semantic maps from dataset A to images from dataset B. Several challenges 
exist in this task due to the scarcity of the labels, and the heterogeneity of the data (from different 
patients and modalities).
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/ctvsmri.png)

## Overview

This repository is about part of my master research project (Forschungsarbeit), which aims at generating realistic looking medical images from semantic label maps. 
In addition, many different images can be generated from any given label map by simply resampling a noise vector.
We implemented [Oasis](https://arxiv.org/abs/2012.04781)-generator, which is based on SPADE and Wavelet-discriminator. 
This repo is supervised paired image synthesis, the first step towards our final goal, using CT labels to generate CT images.
Our final model for unpaired image synthesis is still in progress and will be released soon!


## Setup
First, clone this repository:
```
git clone https://github.com/TWWinde/Medical-Images-Synthesis.git
cd Medical-Images-Synthesis
```

The basic requirements are PyTorch and Torchvision.
```
conda env create MIS
source activate MIS
```
## Datasets

We implement our models based on [AutoPET](https://autopet.grand-challenge.org), which is used for paired supervised model(this repo), and [SynthRAD2023](https://synthrad2023.grand-challenge.org), which is used for unpaired unsupervised model.

## Input Pipeline
For medical images, the pre-processing is of great importance.
Execute ```dataloaders/generate_2d_images.py```to transfer 3d niffti images to slices(2d labels and RGB images).
implementing ```remove_background```function can remove the useless artifacts from medical equipment 
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/WechatIMG3102.png)
The script above results in the following folder structure.

```
data_dir
├── train
|     ├──images
|     └── labels                 
├── test
|     ├──images 
|     └── labels
└── val
      ├──images
      └── labels
```

## Training the model

To train the model, execute the training scripts through ```sbatch batch.sh``` . 
In these scripts you first need to specify the path to the data folder. 
Via the ```--name``` parameter the experiment can be given a unique identifier. 
The experimental results are then saved in the folder ```./checkpoints```, where a new folder for each run is created with the specified experiment name. 
You can also specify another folder for the checkpoints using the ```--checkpoints_dir``` parameter.
If you want to continue training, start the respective script with the ```--continue_train``` flag. 
Have a look at ```config.py``` for other options you can specify.  
Training on 1 NVIDIA A5000 (32GB) is recommended.


## Testing the model

To test a trained model, execute the testing scripts in the ```scripts``` folder. The ```--name``` parameter 
should correspond to the experiment name that you want to test, and the ```--checkpoints_dir``` should the folder 
where the experiment is saved (default: ```./checkpoints```). These scripts will generate images from a pretrained model 
in ```./results/name/```.


## Measuring Metrics

The FID, PIPS, PSNR, RMSE and SSIM are computed on the fly during training, using the popular PyTorch implementation from https://github.com/mseitzer/pytorch-fid. 
At the beginning of training, the inception moments of the real images are computed before the actual training loop starts. 
How frequently the FID should be evaluated is controlled via the parameter ```--freq_fid```, which is set to 5000 steps by default.
The inception net that is used for FID computation automatically downloads a pre-trained inception net checkpoint. 
The Alex net that is used for PIPs computation automatically downloads a pre-trained Alex net checkpoint. 
The results are ploted automatically and shown below.
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/metrics.png)
In oder to compute MIoU, we use the powerful segmentation benchmark--nnUnet. We trained on AutoPET 2d slices and our validation Dice reached 0.78.
The checkpoints for the pre-trained segmentation model is available [here](). For the major classed, the MIoU are more the 0.7. The code of nnUnet id loacted
in my another [repo](https://github.com/TWWinde/nnUNet). After configuration, you can simply execute ```utils/miou_folder/nnunet_segment.py```
to compute the MIoU.

## Pretrained models

The checkpoints for the pre-trained models are available [here]() as zip files. Copy them into the checkpoints folder (the default is ```./checkpoints```, 
create it if it doesn't yet exist) and unzip them. The folder structure should be  

You can generate images with a pre-trained checkpoint via ```test.py```:
```
python test.py --dataset_mode medical --name medical_pretrained \
--dataroot path_to/autopet
```
This script will create a folder named ```./results``` in which the resulting images are saved.

If you want to continue training from this checkpoint, use ```train.py``` with the same ```--name``` parameter and add ```--continue_train --which_iter best```.
## Citation
If you use this work please cite
```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={2024}
}   
```
## Results

The generated images of our model are shown below: 
(From left to right, first images are labels, last images are ground_truth, the images in between are generated images with different random input noise):
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/combined_gerneated1.png)
This is the first edition of the model, which are not rewarding as the shape of the generated images vary a lot, the shape consistency is not 
good enough, especially at the boundary. So we pre-process the input images to remove artifacts from medical equipment(as shown in input pipeline above)
and use Mask Loss to enhance shape consistency. The basic idea is very straightforward and shown below.
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/maskloss.png)
After implementation:
![img.png](https://github.com/TWWinde/Medical-Images-Synthesis/blob/main/assert/combined_generated2.png)





## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.
Write to one of the following email addresses, and maybe put one other in the cc:

twwinde@gmail.com  
st180408@stud.uni-stuttgart.de


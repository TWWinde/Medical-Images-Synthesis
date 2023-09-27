from numpy import generic
from torchvision.transforms import functional
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
from utils.fid_scores import fid_pytorch
from utils.drn_segment import drn_105_d_miou as drn_105_d_miou
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import os
from utils.Metrics import metrics

generate_images = True
compute_miou_generation = False
compute_fid_generation = False
compute_miou_segmentation_network = False
compute_metrics = False

from models.generator import WaveletUpsample,InverseHaarTransform,HaarTransform,WaveletUpsample2
wavelet_upsample = WaveletUpsample()

# from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
# xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
# ifm = DWTInverse(mode='zero', wave='db3')




from utils.utils import tens_to_im
import numpy as np
from torch.autograd import Variable

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)


from collections import namedtuple


#--- read options ---#
opt = config.read_arguments(train=False)
print(opt.phase)

#--- create dataloader ---#
_,_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
#image_saver = utils.results_saver(opt)
image_saver = utils.results_saver_for_test(opt)
#--- create models ---#
model = models.Unpaired_model(opt)
#model = models.Unpaired_model_cycle(opt)
model = models.put_on_multi_gpus(model, opt)
utils.load_networks(opt, model)
model.eval()

mae = []
mse = []



if generate_images :
    #--- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):
        groundtruth, label = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None).cpu().detach()
        image_saver(label, generated, groundtruth, data_i["name"])




        #generated1 = model(None, label, "generate", None).cpu().detach()
        #generated2 = model(None, label, "generate", None).cpu().detach()
        #generated3 = model(None, label, "generate", None).cpu().detach()
        #generated4 = model(None, label, "generate", None).cpu().detach()
        #plt.imshow(tens_to_im(generated[0]))
        #downsampled = torch.nn.functional.interpolate(generated,scale_factor = 0.5)
        #plt.figure()
        #downsampled_wavlet = HaarTransform(3,four_channels=False,levels=1)(downsampled)
        #downsampled_wavlet = HaarTransform(in_channels=3,four_channels=False)(downsampled)

        #upsampled_wavelet = WaveletUpsample2()(downsampled_wavlet)
        #plt.imshow(tens_to_im(downsampled_wavlet[0]))
        #upsampled = InverseHaarTransform(3,four_channels=False,levels=2)(upsampled_wavelet)
        # error = tens_to_im(upsampled[0]-generated[0])-0.5
        #error = tens_to_im(torch.nn.functional.interpolate(torch.nn.functional.interpolate(generated[0],scale_factor = 0.5),scale_factor = 2)-generated[0])-0.5
        #mae.append(np.absolute(error).mean())
        #mse.append(np.sqrt(np.multiply(error,error).mean()))
        #plt.figure()
        #plt.imshow(tens_to_im(upsampled[0]-generated[0]))
        #plt.show()
        #image_saver(label, generated1, generated2, generated3, generated4, groundtruth, data_i["name"])

#print(np.array(mae).mean())
#print(np.array(mse).mean())



'''print(drn_105_d_miou(opt.results_dir,opt.name,'latest'))
print(drn_105_d_miou(opt.results_dir,opt.name,'20000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'40000'))
print(drn_105_d_miou(opt.results_dir,opt.name,'60000'))'''


if compute_miou_generation :
    print(drn_105_d_miou(opt.results_dir,opt.name,opt.ckpt_iter))
else :
    np_file = np.load(os.path.join(opt.checkpoints_dir,opt.name,'MIOU',"miou_log.npy"))
    first = list(np_file[0, :])
    sercon_miou = list(np_file[1, :])
    #first.append(epoch)
    #sercon.append(cur_fid)
    np_file = [first, sercon_miou]
    print('max miou score is :')
    if opt.ckpt_iter == 'latest' :
        print(sercon_miou[-1])
    elif opt.ckpt_iter == 'best' :
        print(np.max(sercon_miou))
    else :
        print(sercon_miou[first.index(float(opt.ckpt_iter))])

if compute_fid_generation :
    fid_computer = fid_pytorch(opt, dataloader_val)
    fid_computer.fid_test(model)
else :
    np_file = np.load(os.path.join(opt.checkpoints_dir,opt.name,'FID',"fid_log.npy"))
    first = list(np_file[0, :])
    sercon = list(np_file[1, :])
    #first.append(epoch)
    #sercon.append(cur_fid)
    np_file = [first, sercon]
    #print('fid score is :')
    if opt.ckpt_iter == 'latest' :
        print(sercon[-1])
    elif opt.ckpt_iter == 'best' :
        print('min fid : ',np.min(sercon))
        index = sercon.index(np.min(sercon))
        #print(len(sercon))
        #print(len(sercon_miou))
        #print(index)
        print('miou : ',sercon_miou[index])

    else :
        print(sercon[first.index(float(opt.ckpt_iter))])





if compute_miou_segmentation_network :
    hist = np.zeros((35, 35))
    for i, data_i in tqdm(enumerate(dataloader_val)):
        image, label = models.preprocess_input(opt, data_i)
        generated_image = model(image, label, "generate", None)
        #plt.imshow(generated_plot)
        #plt.figure()
        generated = model(image, label, "segment_real", None)
        """generated_entropy = generated[0].cpu()-1
        pixel_entropy = torch.zeros(generated_entropy.size()[1:])
        generated_entropy = generated_entropy.permute((1,2,0))
        generated_entropy = Categorical(logits=generated_entropy).entropy().detach()"""
        '''plt.imshow(generated_entropy)'''

        """for j,row in enumerate(pixel_entropy) :
            for k,_ in enumerate(row) :
                pixel_entropy[j,k] = Categorical(probs= generated_entropy[:,j,k]).entropy()
        #generated_plot = torch.argmax(generated, 1).cpu()
        plt.imshow(pixel_entropy)"""
        generated_plot = torch.argmax(generated,1)[0].cpu()-1
        """original_label = torch.argmax(label,1)[0].cpu()
        error_plot = generated_plot != original_label"""
        '''    plt.figure()
        plt.imshow(error_plot.float())
    
        plt.figure()
        plt.imshow(tens_to_im(generated_image[0]))'''

        """plt.figure()
        plt.imshow(tens_to_im(image.cpu()[0]))
        generated = model(None, label, "generate", None)
        image_saver(label, generated, data_i["name"])
        plt.figure()
        plt.imshow(tens_to_im(generated.cpu()[0]))"""

        hist += fast_hist(generated_plot.flatten().numpy(), torch.argmax(label,1).flatten().cpu().numpy(), 35)

        ious = per_class_iu(hist)
        reduced_ius = np.zeros(19)
        for j,iu in enumerate(ious):
            for gt_label in labels :
                if gt_label.id == j and gt_label.trainId != 255 :
                    reduced_ius[gt_label.trainId] = iu

        print('===> mAP {mAP:.3f}'.format(
            mAP=round(np.nanmean(reduced_ius) * 100, 2)),end="\r")



        if (i % 100) == 0 :
            plt.figure()
            plt.imshow(hist, cmap='hot')
            plt.show()

    ious = per_class_iu(hist) * 100
    reduced_ius = np.zeros(19)
    for i, iu in enumerate(ious):
        for gt_label in labels:
            if gt_label.id == i and gt_label.trainId != 255:
                reduced_ius[gt_label.trainId] = iu

    print(' '.join('{:.03f}'.format(i) for i in reduced_ius))
    print(' '.join('{:.03f}'.format(i) for i in ious))
    print(round(np.nanmean(reduced_ius), 2))
    print(round(np.nanmean(ious), 2))








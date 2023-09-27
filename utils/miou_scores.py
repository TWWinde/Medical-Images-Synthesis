import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import models.models as models
from utils.fid_folder.inception import InceptionV3
import matplotlib.pyplot as plt
from utils import utils
from utils.miou_folder.nnunet_segment import get_predicted_label, compute_miou


# --------------------------------------------------------------------------#
# This code is an adapted version of https://github.com/mseitzer/pytorch-fid
# --------------------------------------------------------------------------#



class miou_pytorch():
    def __init__(self, opt, dataloader_val):
        self.opt = opt
        self.val_dataloader = dataloader_val
        self.best_miou = 0
        self.path_to_save = os.path.join(self.opt.checkpoints_dir, self.opt.name, "MIOU")
        Path(self.path_to_save).mkdir(parents=True, exist_ok=True)

    def compute_miou(self, netG, netEMA,model = None,current_iter = 'latest'):
        image_saver = utils.results_saver_mid_training(self.opt,str(current_iter))
        netG.eval()
        if not self.opt.no_EMA:
            netEMA.eval()
        with torch.no_grad():
            n=1
            for i, data_i in enumerate(self.val_dataloader):
                image, label = models.preprocess_input(self.opt, data_i)
                edges = model.module.compute_edges(image)
                if self.opt.no_EMA:
                    generated = netG(label,edges=edges)
                else:
                    generated = netEMA(label,edges=edges)
                image_saver(label, generated, data_i["name"])
                n+=1
                if n >100:
                    break
            if self.opt.dataset_mode == "medicals" or self.opt.dataset_mode == "medicals_no_3d_noise":
                #get_predicted_label(self.opt, current_iter)
                pred_folder = os.path.join(self.opt.results_dir, self.opt.name, str(current_iter), 'segmentation')
                gt_folder = os.path.join(self.opt.results_dir, self.opt.name, str(current_iter), 'label')
                #answer = compute_miou(self.opt, pred_folder, gt_folder)

        netG.train()
        if not self.opt.no_EMA:
            netEMA.train()
        return 1#answer


    def update(self, model, cur_iter):
        print("--- Iter %s: computing MIOU ---" % (cur_iter))
        cur_miou = self.compute_miou(model.module.netG, model.module.netEMA,model,cur_iter)
        self.update_logs(cur_miou, cur_iter)
        print("--- MIOU at Iter %s: " % cur_iter, "{:.2f}".format(cur_miou))
        if cur_miou < self.best_miou:
            self.best_miou = cur_miou
            is_best = True
        else:
            is_best = False
        return is_best

    def update_logs(self, cur_miou, epoch):
        try :
            np_file = np.load(self.path_to_save + "/miou_log.npy")
            first = list(np_file[0, :])
            sercon = list(np_file[1, :])
            first.append(epoch)
            sercon.append(cur_miou)
            np_file = [first, sercon]
        except:
            np_file = [[epoch], [cur_miou]]

        np.save(self.path_to_save + "/miou_log.npy", np_file)

        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.savefig(self.path_to_save + "/plot_miou", dpi=600)
        plt.close()


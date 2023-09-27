import torch
import numpy as np
import random
import time
import os
import models.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter  = (start_iter + 1) %  dataset_size
    return start_epoch, start_iter


def remove_background(image):
    # Apply Gaussian blur to the image (optional)
    binary_image = np.where(image > 45, 1, 0)
    binary_image = binary_image.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    result_image = np.zeros_like(binary_image)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 1000:
            result_image[labels == label] = 255
            #result_image = morphology.binary_opening(result_image, )
            #result_image = morphology.remove_small_holes(result_image,area_threshold=8)
    result_image = result_image.astype(np.uint8)
    result_image = cv2.multiply(image, result_image*255)

    return result_image


class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        self.path_combined = os.path.join(path, "combined")
        os.makedirs(self.path_combined, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated1, generated2, generated3, generated4, groundtruth, name):
        assert len(label) == len(generated1)
        for i in range(len(label)):
            im_label = tens_to_lab_color(label[i], self.num_cl)
            im_image1 = tens_to_im(generated1[i]) * 255
            im_image2 = tens_to_im(generated2[i]) * 255
            im_image3 = tens_to_im(generated3[i]) * 255
            im_image4 = tens_to_im(generated4[i]) * 255
            im_image5 = tens_to_im(groundtruth[i]) * 255
            #out.clamp(0, 1)
            im_image1 = im_image1.astype(np.uint8)
            im_image2 = im_image2.astype(np.uint8)
            im_image3 = im_image3.astype(np.uint8)
            im_image4 = im_image4.astype(np.uint8)
            im_image5 = im_image5.astype(np.uint8)

            combined_image = self.combine_images(im_label, im_image1, im_image2, im_image3, im_image4, im_image5)
            self.save_combined_image(combined_image, name[i])

    def combine_images(self, im_label, im_image1, im_image2, im_image3, im_image4, im_image5):
        width, height = im_label.shape[1], im_label.shape[0]
        combined_image = Image.new("RGB", (width * 6, height))
        combined_image.paste(Image.fromarray(im_label), (0, 0))
        combined_image.paste(Image.fromarray(im_image1), (width, 0))
        combined_image.paste(Image.fromarray(im_image2), (width * 2, 0))
        combined_image.paste(Image.fromarray(im_image3), (width * 3, 0))
        combined_image.paste(Image.fromarray(im_image4), (width * 4, 0))
        combined_image.paste(Image.fromarray(im_image5), (width * 5, 0))
        return combined_image

    def save_combined_image(self, combined_image, name):
        combined_image.save(os.path.join(self.path_combined, name.split("/")[-1]).replace('.jpg', '.png'))


class results_saver_for_test():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name,'test')
        self.path_label = os.path.join(path, "label")
        self.path_generated = os.path.join(path, "generated")
        self.path_groundtruth = os.path.join(path, "groundtruth")
        self.path_segmentation = os.path.join(path, "segmentation")
        self.path_to_save = {"label": self.path_label, "generated": self.path_generated, "groundtruth": self.path_groundtruth}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_generated, exist_ok=True)
        os.makedirs(self.path_segmentation, exist_ok=True)
        os.makedirs(self.path_groundtruth, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, groundtruth, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            name_label = name[i].split("/")[-1].replace('.jpg', '.png')
            name_image = name_label.split(".")[0] + '_0000.' + name_label.split(".")[-1]
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name_label)
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "generated", name_image)
            im = tens_to_im(groundtruth[i]) * 255
            self.save_im(im, "groundtruth", name_image)

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name))


class results_saver_mid_training():
    def __init__(self, opt, current_iteration):
        path = os.path.join(opt.results_dir, opt.name, current_iteration)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_segmentation = os.path.join(path, "segmentation")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        os.makedirs(self.path_segmentation, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            name_label = name[i].split("/")[-1].replace('.jpg', '.png')
            name_image = name_label.split(".")[0] + '_0000.' + name_label.split(".")[-1]
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name_label)
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name_image)

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name))

class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt):
        if opt.model_supervision == 2:
            self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        elif opt.model_supervision ==1:
            self.name_list = ["sup_G_Du",
                              "sup_G_D",
                              "sup_VGG",
                              "sup_G_feat_match",
                              "sup_D_fake",
                              "sup_D_real",
                              "sup_D_LM",
                              "sup_Du_fake",
                              "sup_Du_real",
                              "un_G_D",
                              "un_VGG",
                              "un_G_Du",
                              "un_edge",
                              "un_Du_fake",
                              "un_Du_real",
                              "un_Du_regularize",
                              "sup_Du_regularize"]
        else:
            self.name_list = ["Generator", "Vgg", "GAN","edge","mask",'featMatch',"D_fake", "D_real", "LabelMix","Du_fake","Du_real","Du_regularize"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        print(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with torch.no_grad():
        for key in model.module.netEMA.state_dict():
            model.module.netEMA.state_dict()[key].data.copy_(
                model.module.netEMA.state_dict()[key].data * opt.EMA_decay +
                model.module.netG.state_dict()[key].data   * (1 - opt.EMA_decay)
            )
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label = models.preprocess_input(opt, data_i)
                fake = model.module.netEMA(label,edges = model.module.compute_edges(image))
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        torch.save(model.module.netG.state_dict(), path+'/%s_G.pth' % ("latest"))
        try:
            torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("latest"))
        except:
            pass
        try:
            torch.save(model.module.netDu_image.state_dict(), path + '/%s_Du_image.pth' % ("latest"))
            torch.save(model.module.netDu_label.state_dict(), path + '/%s_Du_label.pth' % ("latest"))
        except:
            pass
        try:
            torch.save(model.module.netDu.state_dict(), path + '/%s_Du.pth' % ("latest"))
        except:
            pass
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("latest"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        torch.save(model.module.netG.state_dict(), path+'/%s_G.pth' % ("best"))
        try:
            torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("best"))
        except:
            pass
        try:
            torch.save(model.module.netDu_image.state_dict(), path + '/%s_Du_image.pth' % ("best"))
            torch.save(model.module.netDu_label.state_dict(), path + '/%s_Du_label.pth' % ("best"))
        except:
            pass
        try:
            torch.save(model.module.netDu.state_dict(), path + '/%s_Du.pth' % ("best"))
        except:
            pass
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("best"))
        with open(os.path.join(opt.checkpoints_dir, opt.name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        torch.save(model.module.netG.state_dict(), path+'/%d_G.pth' % (cur_iter))
        try:
            torch.save(model.module.netD.state_dict(), path+'/%d_D.pth' % (cur_iter))
        except:
            pass
        try:
            torch.save(model.module.netDu_image.state_dict(), path+'/%d_Du_image.pth' % (cur_iter))
            torch.save(model.module.netDu_label.state_dict(), path+'/%d_Du_label.pth' % (cur_iter))
        except:
            pass
        try:
            torch.save(model.module.netDu.state_dict(), path + '/%d_Du.pth' % (cur_iter))
        except:
            pass
        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%d_EMA.pth' % (cur_iter))


def load_networks(opt, model):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    try:
        checkpoint1 = torch.load(os.path.join(path, 'best_G.pth'))
        model.module.netG.load_state_dict(checkpoint1)
        checkpoint2 = torch.load(os.path.join(path, 'best_D.pth'))
        model.module.netG.load_state_dict(checkpoint2)
        print('checkpoints successfully loaded')
    except:
        print('checkpoints dont exist')
        pass
    try:
        checkpoint5 = torch.load(os.path.join(path, 'best_Du_image.pth'))
        model.module.netG.load_state_dict(checkpoint5)
        checkpoint6 = torch.load(os.path.join(path, 'best_Du_label.pth'))
        model.module.netG.load_state_dict(checkpoint6)
    except:
        pass
    try:
        checkpoint3 = torch.load(os.path.join(path, 'best_Du.pth'))
        model.module.netG.load_state_dict(checkpoint3)
    except:
        pass
    if not opt.no_EMA:
        try:
            checkpoint4 = torch.load(os.path.join(path, 'best_EMA.pth'))
            model.module.netG.load_state_dict(checkpoint4)
        except:
            pass



class test_image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.results_dir, opt.name, "images")+"/"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        edges = model.module.compute_edges(image)
        with torch.no_grad():
            model.eval()
            fake = model.module.netG(label,edges=edges)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                fake = model.module.netEMA(label,edges=edges)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab_color(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images")+"/"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        edges = model.module.compute_edges(image)
        with torch.no_grad():
            model.eval()
            fake = model.module.netG(label,edges=edges)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                fake = model.module.netEMA(label, edges=edges)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab_color(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = GreyScale(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

def tens_to_lab_color(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def GreyScale(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = label
        color_image[1][mask] = label
        color_image[2][mask] = label
    return color_image

def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()  # [38, 256, 256]
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)  # [1, 256, 256]

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 39:
        cmap = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81),(50, 80, 100), (0, 100, 230), (119, 60, 50), (70, 40, 142),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142), (150, 250, 90), (0, 153, 140), (119, 11, 32), (0, 0, 142),(150, 250, 90), (0, 153, 140)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap






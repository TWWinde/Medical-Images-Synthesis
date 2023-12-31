import torch


def get_dataset_name(mode):
    if mode == "medicals":
        return "MedicalImagesDataset"
    if mode == "medicals_no_3d_noise":
        return "MedicalImagesDataset"
    else:
        return "MedicalImagesDataset"
        #ValueError("There is no such dataset regime as %s" % mode)

def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders."+dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_supervised = file.__dict__[dataset_name].__dict__[dataset_name](opt,for_metrics = False ,for_supervision = True)
    dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False)
    dataloader_supervised = torch.utils.data.DataLoader(dataset_supervised, batch_size = opt.batch_size, shuffle = True, drop_last=True)
    return dataloader_train,dataloader_supervised, dataloader_val
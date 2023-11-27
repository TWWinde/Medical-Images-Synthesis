import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


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

def get_2d_images(ct_path, label_path):
    n = 0
    for i in range(int(len(ct_path) * 0.9)):
        break
        nifti_img = nib.load(ct_path[i])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[i])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/train/images/slice_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/train/labels/slice_{n}.png', seg_slice)
                n += 1

    print("finished train data set")
    n = 0
    x = 0
    for j in range(int(len(ct_path) * 0.9), int(len(ct_path) * 0.95)):
        nifti_img = nib.load(ct_path[j])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[j])
        seg_3d = nifti_seg.get_fdata()
        x += 1
        n = 0
        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/test1/images/slice_{x}_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/test1/labels/slice_{x}_{n}.png', seg_slice)
                n += 1

    print("finished test data set")
    n = 0
    for k in range(int(len(ct_path) * 0.95), len(ct_path)):
        break
        nifti_img = nib.load(ct_path[k])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[k])
        seg_3d = nifti_seg.get_fdata()

        for z in range(seg_3d.shape[2]):
            seg_slice = seg_3d[:, :, z]
            seg_slice = seg_slice[72:328, 65:321]
            seg_slice = seg_slice.astype(np.uint8)
            img_slice = img_3d[:, :, z]
            img_slice = img_slice[72:328, 65:321]

            if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
                image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
                image = remove_background(image)
                image = Image.fromarray(image)
                image = image.convert('RGB')
                image.save(f'/misc/data/private/autoPET/data_nnunet/val/images/slice_{n}.png')
                cv2.imwrite(f'/misc/data/private/autoPET/data_nnunet/val/labels/slice_{n}.png', seg_slice)
                n += 1

    print("finished validation data set")

def list_images(path):
    ct_path = []
    label_path = []
    # read autoPET files names
    names = os.listdir(path)
    ct_names = list(filter(lambda x: x.endswith('0001.nii.gz'), names))

    for i in range(len(ct_names)):
        ct_path.append(os.path.join(path, ct_names[i]))
        label_path.append(os.path.join(path, ct_names[i].replace('0001.nii.gz', '0002.nii.gz')))

    return ct_path, label_path



''''
os.makedirs('/misc/data/private/autoPET/train3/CT1', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/train3/SEG1', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/test3/CT1', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/test3/SEG1', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/val3/CT1', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/val3/SEG1', exist_ok=True)
'''
os.makedirs('/misc/data/private/autoPET/data_nnunet/test1/images/', exist_ok=True)
os.makedirs('/misc/data/private/autoPET/data_nnunet/test1/labels/', exist_ok=True)
path_imagesTr = "/misc/data/private/autoPET/imagesTr"
root_dir = "/misc/data/private/autoPET/"
test_path = '/misc/data/private/autoPET/train/SEG'

ct_paths, label_paths = list_images(path_imagesTr)
#unique_value = set()
# for item in label_paths:
#     nifti_seg = nib.load(item)
#     seg_3d = nifti_seg.get_fdata()
#     flat_data = seg_3d.flatten().tolist()
#     for value in flat_data:
#         unique_value.add(value)

#for item in sorted(os.listdir(test_path)):
#    image_path = os.path.join(test_path, item)
#    image = cv2.imread(image_path)
 #   flat_data = image.flatten().tolist()
 #   for value in flat_data:
 #       unique_value.add(value)

#print(unique_value)
#print('number of classes:',len(unique_value))


get_2d_images(ct_paths, label_paths)



def preprocess_input_images(path):
    for item in sorted(os.listdir(path)):
        image_path = os.path.join(path, item)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        result_image = remove_background(image)
        cv2.imwrite(image_path.replace('CT', 'CT1'), result_image)



#preprocess_input_images('/misc/data/private/autoPET/train3/CT')
#preprocess_input_images('/misc/data/private/autoPET/test3/CT')
#preprocess_input_images('/misc/data/private/autoPET/val3/CT')

def rename_copy(root_dir):

    name = ['train3','test3','val3']
    n = 1
    #for i in range(3):
        #for item in sorted(os.listdir(os.path.join(root_dir, name[i], 'SEG'))):
            #original_file_path = os.path.join(root_dir, name[i], 'SEG', item)
            #new_filename = f'body_{n:06}.nii.gz'
            #destination_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_AutoPET/labelsTr'
            #new_file_path = os.path.join(destination_folder, new_filename)
            #shutil.copyfile(original_file_path, new_file_path)
            #n+=1


    print('label finished')
    n = 1
    for i in range(3):
        for item in sorted(os.listdir(os.path.join(root_dir, name[i], 'CT1'))):
            original_file_path = os.path.join(root_dir, name[i], 'CT1', item)
            new_filename = f'body_{n:06}_0001.nii.gz'
            destination_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/Dataset521_AutoPET/imagesTr'
            new_file_path = os.path.join(destination_folder, new_filename)
            png_image = cv2.imread(original_file_path)
            nifti_image = nib.Nifti1Image(png_image, np.eye(4))  # 这里使用单位矩阵作为仿射矩阵
            nib.save(nifti_image, new_file_path)
            n+=1
    print('images finished')

#rename_copy(root_dir)
'''
from typing import Tuple

from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didnt like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)
    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)




output_folder = '/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Dataset001_AutoPET'
name = {1: 'CT'}
labels = {'background': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
           '10': 10,
           '11': 11,
           '12': 12,
            '13': 13,
            '14': 14,
            '15': 15,
            '16': 16,
            '17': 17,
            '18': 18,
           '19': 19,
           '20': 20,
            '21': 21,
            '22': 22,
            '23': 23,
            '24': 24,
            '25': 25,
            '26': 26,
            '27': 27,
            '28': 28,
            '29': 29,
            '30': 30,
            '31': 31,
            '32': 32,
            '33': 33,
            '34': 34,
            '35': 35,
            '36': 36
          }


#generate_dataset_json(output_folder, name, labels, 352724, ".nii.gz",dataset_name='AutoPET', )



#export nnUNet_raw="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_raw"
#export nnUNet_preprocessed="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_preprocessed"
#export nnUNet_results="/no_backups/s1449/nnUNetFrame/DATASET/nnUNet_results"



print('finished')
'''
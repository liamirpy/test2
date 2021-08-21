import nibabel as nib
import numpy as np
from time import sleep
from lungmask import mask
import SimpleITK as sitk


def rotate_output(segmentation):
    rot1 = np.zeros((segmentation.shape[2], segmentation.shape[1], segmentation.shape[0]))
    for i in range(segmentation.shape[0]):
        rot1[:, :, i] = np.rot90(segmentation[i, :, :])

    rot2 = np.zeros((segmentation.shape[2], segmentation.shape[1], segmentation.shape[0]))
    for i in range(segmentation.shape[0]):
        rot2[:, :, i] = np.flipud(rot1[:, :, i])

    result=rot2
    return result



INPUT=input('Please enter folder name : \n')
sleep(1)
original=nib.load(INPUT)

input_image = sitk.ReadImage(INPUT)
hole_part = mask.apply(input_image)
hole_result=rotate_output(hole_part)

img = nib.Nifti1Image(hole_result, original.affine)
hole_result_name=INPUT.split('.nii')[0] + '_hole.nii'
nib.save(img,hole_result_name)
sleep(3)
########
model = mask.get_model('unet','LTRCLobes')
health_part = mask.apply(input_image, model)
health_result=rotate_output(health_part)
img = nib.Nifti1Image(hole_result, original.affine)
health_result_name=INPUT.split('.nii')[0] + '_health.nii'
nib.save(img,health_result_name)
sleep(3)



import os.path
import nibabel as nib
import numpy as np
import h5py
import json
from medutils.mri import ifft2c, rss


def get_yshift(hf_file):
    """Get the y_shift to be applied on reconstructed raw images."""

    tmp = hf_file['mrecon_header']['Parameter']['YRange'][()]
    if len(np.unique(tmp[0])) > 1 or len(np.unique(tmp[1])) > 1:
        print('Error: different y shifts for different echoes!')
    return -int((tmp[0, 0] + tmp[1, 0]) / 2)


def load_raw_data(file_path):
    """Load raw data from the T2*-MOVE dataset."""

    with h5py.File(file_path, "r") as hf_file:
        raw_data = hf_file['kspace'][()]
        sens_maps = hf_file['sens_maps'][()]
        y_shift = get_yshift(hf_file)

    return raw_data, sens_maps, y_shift


def pad_sensitivity_maps(sens_maps, kspace_shape):
    """Pad coil sensitivity maps to have same shape as images."""

    pad_width = ((0, 0), (0, 0), (0, 0), (0, 0),
                 (int((kspace_shape[-1] - sens_maps.shape[-1]) / 2),
                  int((kspace_shape[-1] - sens_maps.shape[-1]) / 2))
                 )
    sens_maps = np.pad(sens_maps, pad_width, mode='constant')
    return np.nan_to_num(sens_maps / rss(sens_maps, 2)[:, None])


def remove_readout_oversampling(data, nr_lines):
    """Remove readout oversampling."""

    return data[..., nr_lines:-nr_lines]


def compute_coil_combined_reconstructions(kspace, sens_maps,
                                          y_shift, remove_oversampling=True):
    """Compute coil combined reconstructions."""

    coil_imgs = ifft2c(kspace)
    coil_imgs = np.roll(coil_imgs, shift=y_shift, axis=-2)
    sens_maps = pad_sensitivity_maps(sens_maps, kspace.shape)
    img_cc = np.sum(coil_imgs * np.conj(sens_maps), axis=2)
    if remove_oversampling:
        img_cc = remove_readout_oversampling(img_cc,
                                             int(img_cc.shape[-1] / 4))
    return img_cc


def load_coil_combined_reconstruction(file_path):
    """Load the coil-combined reconstruction from the T2*-MOVE dataset."""

    with h5py.File(file_path, "r") as hf_file:
        img_cc = hf_file['reconstruction'][()]
        nii_header = {}
        for key in hf_file['nifti_header'].keys():
            nii_header[key] = hf_file['nifti_header'][key][()]

    return img_cc, nii_header


def load_reference_mask(file_path):
    """Load the reference exclusion mask for the motion-corrupted acquisition."""

    if os.path.exists(file_path):
        tmp = np.loadtxt(file_path, unpack=True).T
        # shift to match the correct timing:
        tmp = np.roll(tmp, 3, axis=1)
        tmp[:, 0:3] = 1
        # mask_timing = np.take(tmp, idx, axis=0)
        return tmp

    else:
        print(f"Reference mask file {file_path} does not exist.")
        return None


def load_segmentation(file_path, binary=True):
    """Load mask from nii file."""

    mask = nib.load(file_path).get_fdata()[10:-10][::-1, ::-1, :]
    mask = np.rollaxis(mask, 2, 0)
    if binary:
        mask = np.where(mask < 0.5, 0, 1)

    return mask


def load_motion_data(file_path):
    """Load motion data from a JSON file."""

    with open(os.path.join(file_path), 'r') as f:
        data = json.load(f)

    data.pop("RMS_displacement")
    data.pop("max_displacement")
    data.pop("motion_free")

    return data

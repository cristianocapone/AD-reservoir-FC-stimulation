#!/usr/bin/env python3
"""
Motion correction script using dipy for rigid registration.
Applies motion correction to BOLD time series using the middle volume as reference.
"""

import nibabel as nib
import numpy as np
from dipy.align import affine_registration, rigid
from dipy.align.transforms import RigidTransform3D
import os
import sys

def motion_correct_bold(bold_file, output_file=None, reference_volume=None):
    """
    Perform motion correction on BOLD time series.

    Parameters:
    - bold_file: path to BOLD NIfTI file
    - output_file: path for output corrected file (default: add '_mc' suffix)
    - reference_volume: index of reference volume (default: middle)
    """
    print(f"Loading BOLD file: {bold_file}")

    # Load the BOLD image
    img = nib.load(bold_file)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    n_volumes = data.shape[3]
    print(f"Found {n_volumes} volumes")

    # Select reference volume (middle by default)
    if reference_volume is None:
        reference_volume = n_volumes // 2

    print(f"Using volume {reference_volume} as reference")

    ref_data = data[:, :, :, reference_volume]
    ref_img = nib.Nifti1Image(ref_data, affine)

    # Initialize corrected data array
    corrected_data = np.zeros_like(data)

    # Motion parameters (translations and rotations)
    motion_params = []

    # Register each volume to the reference
    for i in range(n_volumes):
        if i == reference_volume:
            # Reference volume stays the same
            corrected_data[:, :, :, i] = ref_data
            motion_params.append([0, 0, 0, 0, 0, 0])  # No motion
            continue

        print(f"Processing volume {i+1}/{n_volumes}")

        moving_data = data[:, :, :, i]
        moving_img = nib.Nifti1Image(moving_data, affine)

        # Perform rigid registration
        from dipy.align import rigid
        reg_affine, transformed = rigid(moving_img, ref_img)

        # Extract motion parameters from affine
        # Affine matrix to rotations and translations
        # For rigid transform, we can decompose
        # But for simplicity, save the affine matrix
        # To get rotations and translations, we can use scipy.spatial.transform

        from scipy.spatial.transform import Rotation
        # Extract rotation and translation from affine
        rotation_matrix = reg_affine[:3, :3]
        translation = reg_affine[:3, 3]

        # Convert rotation matrix to Euler angles
        rot = Rotation.from_matrix(rotation_matrix)
        euler_angles = rot.as_euler('xyz', degrees=True)  # in degrees

        motion_params.append([
            translation[0], translation[1], translation[2],
            euler_angles[0], euler_angles[1], euler_angles[2]
        ])

        # Apply transformation
        corrected_data[:, :, :, i] = transformed.get_fdata()

    # Save corrected image
    if output_file is None:
        base, ext = os.path.splitext(bold_file)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
        output_file = base + '_mc.nii.gz'

    corrected_img = nib.Nifti1Image(corrected_data, affine, header)
    nib.save(corrected_img, output_file)
    print(f"Saved corrected image to: {output_file}")

    # Save motion parameters
    motion_file = output_file.replace('.nii.gz', '_motion.txt')
    np.savetxt(motion_file, motion_params, header='tx ty tz rx ry rz', comments='')
    print(f"Saved motion parameters to: {motion_file}")

    return output_file, motion_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python motion_correct.py <bold_file>")
        sys.exit(1)

    bold_file = sys.argv[1]
    motion_correct_bold(bold_file)
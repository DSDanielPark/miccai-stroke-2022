import os
from tempfile import mkstemp
import numpy as np
import nibabel as nib
import ants


class AntsRegistration:
    def __init__(self) -> None:
        self.type_of_transform = "Rigid"

        # select type_of_transform
        # "Translation", "Rigid", "Similarity", "QuickRigid", "DenseRigid",
        # "BOLDRigid", "Affine", "AffineFast",
        # "SyN", "SyNRA", "SyNOnly", "SyNCC", "SyNabp", "SyNBold", "SyNBoldAff", "SyNAggro",
    
        # for more infomation of each registration parameter: https://antspyx.readthedocs.io/en/latest/registration.html

    def to_nibabel(image):
        """
        Convert an ANTsImage to a Nibabel image
        """

        fd, tmpfile = mkstemp(suffix=".nii.gz")
        image.to_filename(tmpfile)
        new_img = nib.load(tmpfile)
        os.close(fd)
        # os.remove(tmpfile) ## do not remove tmpfile because nibabel lazy at loading the data.
        return new_img

    def ants_registration(self, nii_file_ath1: str, nii_file_path2: str, save_path: str) -> None:
        """
        Registration 2 nifti image and save
        """
        fixed_image = ants.image_read(nii_file_ath1)
        moving_image = ants.image_read(nii_file_path2)
        moving_image = ants.resample_image(moving_image, fixed_image.shape, 1, 0)
        ants_registration_result = ants.registration(
            fixed=fixed_image, moving=moving_image, type_of_transform=self.type_of_transform
        )
        warped_mov_out = self.to_nibabel(ants_registration_result["warped_mov_out"])
        warped_fix_out = self.to_nibabel(ants_registration_result["warped_fix_out"])
        nib.save(warped_mov_out, os.path.join(save_path, nii_file_ath1.split("/")[-1] + ".nii"))
        nib.save(warped_fix_out, os.path.join(save_path, nii_file_ath1.split("/")[-1] + "fixed.nii"))

        return None 

    def get_rigid_transformed_img(
        self, t1_img: ants.core.ANTsImage, angio_img: ants.core.ANTsImage
    ) -> ants.core.ANTsImage:
        """
        just for Rigid registration of nifti data from same person
        """
        transform = ants.registration(fixed=t1_img, moving=angio_img, type_of_transform="Rigid")
        moved_img = ants.apply_transforms(
            fixed=t1_img, moving=angio_img, transformlist=transform["fwdtransforms"]
        )
        return moved_img
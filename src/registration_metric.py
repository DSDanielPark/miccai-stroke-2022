import numpy as np
import SimpleITK as sitk
import ants
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt


class RegistrationMetric:
    def __init__(self):
        self.ant_neighborhood_radius = 1

    def calculate_registration_metric(self, nii_file_path1: str, nii_file_path2: str) -> dict:
        img1 = sitk.ReadImage(nii_file_path1)
        img2 = sitk.ReadImage(nii_file_path2)

        metric_dict = dict()
        registration_method = sitk.ImageRegistrationMethod()

        for metric in [
            "mutual_info",
            "ANTS_neighborhood_corr",
            "corr",
            "joint_histogram_mutualInfo",
            "mean_square",
        ]:

            if metric == "mutual_info":
                registration_method.SetMetricAsMattesMutualInformation()
            elif metric == "ANTS_neighborhood_corr":
                registration_method.SetMetricAsANTSNeighborhoodCorrelation(
                    self.ant_neighborhood_radius
                )
            elif metric == "corr":
                registration_method.SetMetricAsCorrelation()
            elif metric == "joint_histogram_mutualInfo":
                registration_method.SetMetricAsJointHistogramMutualInformation()
            elif metric == "mean_square":
                registration_method.SetMetricAsMeanSquares()

                # for more infomation check this page 
                # https://simpleitk.org/SPIE2019_COURSE/04_basic_registration.html

            metric_dict[metric] = np.round(registration_method.MetricEvaluate(img1, img2), 4)

        return metric_dict

    def display_images_with_alpha(image_z: int, alpha: np.float_, fixed: ants.core.ANTsImage, moving: ants.core.ANTsImage) -> plt:

        img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
        plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
        plt.axis("off")
        plt.show()

    def visuall_validation(self, nii_file_path1: str, nii_file_path2: str):
        fixed_image = sitk.ReadImage(nii_file_path1)
        moving_image = sitk.ReadImage(nii_file_path2)

        interact(
            self.display_images,
            fixed_image_z=(0, fixed_image.GetSize()[2] - 1),
            moving_image_z=(0, moving_image.GetSize()[2] - 1),
            fixed_npa=fixed(sitk.GetArrayViewFromImage(fixed_image)),
            moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)),
        )
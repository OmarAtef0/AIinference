'''
steps:
- convert dicom to nii
- transformation
- dataset from transformation -> model -> output one hot tensor
- convert model output to diagnosis (str)
'''

import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import SimpleITK as sitk
from nibabel.orientations import aff2axcodes
from monai.data import DataLoader, decollate_batch, CacheDataset
from monai.networks.layers import Norm
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Resized,
)
import argparse
import torch
import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(CNN3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))

        x = self.global_avg_pool(x).view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return self.sigmoid(x)


def convert_dicom_to_nifti(dicom_folder, output_dir):
    """
    Converts a DICOM series to a NIfTI file and returns its orientation.

    Parameters:
        dicom_folder (str): Path to the folder containing DICOM series.
        output_dir (str): Directory to save the converted NIfTI file.

    Returns:
        tuple: (output_nifti_path, orientation_code)
    """
    # Create output filename based on folder name
    output_filename = os.path.basename(os.path.normpath(dicom_folder)) + ".nii.gz"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, output_filename)

    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    if not dicom_series:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    reader.SetFileNames(dicom_series)
    image = reader.Execute()

    # Save NIfTI file
    sitk.WriteImage(image, output_path)

    return output_path

def init_transforms():
    val_transforms = Compose([
        LoadImaged(keys='image'),
        EnsureChannelFirstd(keys='image'), 
        ScaleIntensityd(keys='image'), 
        Resized(keys='image', spatial_size=(128,128,128)),
        ])
    return val_transforms

def load_data(nii_path):
    try:
        data_val = [{"image": nii_path}]
        val_transforms = init_transforms()
        test_ds = CacheDataset(data=data_val, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
        print("Dataloader and dataset done")
        return test_loader
    except Exception as e:
        print(f"Error creating dataset and dataloader: {e}")
        return

def init_model(weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    model = CNN3D(in_channels=1, num_classes=2).to(device)
    
    # Load the state_dict, removing 'module.' prefix if necessary
    state_dict = torch.load(weights_path, map_location=device)
    
    # Check if the model was saved with DataParallel
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    # Load the modified state_dict
    model.load_state_dict(new_state_dict)

    return model, device


def convert_output_to_diagnosis(model_output):
    if model_output[0] == 0:
        diagnosis = "No Caner"
    else:
        diagnosis = "Cancer"
    return diagnosis

def infere(input_path, weights_path):
    # Check if the input is already a NIfTI file
    if input_path.endswith(".nii") or input_path.endswith(".nii.gz"):
        print("Input is already a NIfTI file. Skipping conversion.")
        nii_path = input_path  # Use the original path
    else:
        # Assume it's a DICOM directory and attempt conversion
        try:
            nii_path = convert_dicom_to_nifti(input_path, "temp")
            print("Data converted to NIfTI")
        except Exception as e:
            print(f"Error converting to NIfTI: {e}")
            return

    test_loader = load_data(nii_path)

    try:
        model, device = init_model(weights_path)
        model.eval()
        print("model init done")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    try:
        with torch.no_grad():
            for test_data in (test_loader):
                test_input = test_data["image"].to(device)
                test_output = model(test_input)
                model_output = test_output.argmax(dim=1)
                model_output = model_output.detach().cpu().numpy()
                diagnosis = convert_output_to_diagnosis(model_output)
                return diagnosis
    except Exception as e:
        print(f"Error during inference: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Input DICOM folder path")
    parser.add_argument("weight_path", type=str, help="Path to model weights")
    args = parser.parse_args()
    input_path,weight_path = args.input_path, args.weight_path
    print(args)
    try:
        diagnosis = infere(input_path,weight_path)
        print(diagnosis)
    except Exception as e:
        print(f"Critical error: {e}")

from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from data import initialize_data # data.py in the same folder
from data import data_transforms
from model import Net
import torchvision
import random
import pandas as pd



parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()

state_dict = torch.load(args.model, weights_only=True)
model = Net()
model.load_state_dict(state_dict)
model.eval()

from data import data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale

ground_truth_dir = args.data + '/Test.csv'
ground_truth = pd.read_csv(ground_truth_dir)
ground_truth_dict = dict(zip(ground_truth['Path'], ground_truth['ClassId']))

test_dir = args.data + '/Test'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transforms = [data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center]
output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")

# print("Test directory:", test_dir)
# print("Files in test directory:", os.listdir(test_dir))
# for f in tqdm(os.listdir(test_dir)):
all_test_files = [f for f in os.listdir(test_dir) if 'png' in f]  # Filter only PNG files

# Select 10% of test files
subset_size = int(0.1 * len(all_test_files))
test_subset = random.sample(all_test_files, subset_size)

correct_predictions = 0
total_predictions = 0


# Loop through the 10% subset
for f in tqdm(test_subset):
    if 'png' in f:
        output = torch.zeros([1, 43], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0,len(transforms)):
                data = transforms[i](pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data)
                output = output.add(model(data))

            # pred = output.data.max(1, keepdim=True)[1]
            # file_id = f[0:5]
            # output_file.write("%s,%d\n" % (file_id, pred))
            # Get the predicted class
            pred = output.data.max(1, keepdim=True)[1].item()

            # Extract the ground truth ClassId using the file name
            file_path = 'Test/' + f
            true_class_id = ground_truth_dict[file_path]

            # Compare prediction with ground truth
            if pred == true_class_id:
                correct_predictions += 1

            total_predictions += 1

            # Optionally write predictions to the output file
            output_file.write("%s,%d,%d\n" % (f[:5], pred, true_class_id))

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2017/')
        



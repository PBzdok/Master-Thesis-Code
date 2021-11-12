import numpy as np
import torch
import torchxrayvision as xrv
from captum.attr import visualization as viz
from captum.attr import Occlusion
from torchvision.transforms import transforms

from data import load_rsna_dataset, calculate_rsna_metrics

d_rsna = load_rsna_dataset()
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')

# df, result = calculate_rsna_metrics(model, d_rsna, force=True)
#
# print(f'Accuracy: {result["accuracy"]}')
# print(f'Precision: {result["precision"]}')
# print(f'Recall: {result["recall"]}')
# print(f'F1: {result["f1"]}')

transform = transforms.Compose([
    transforms.ToTensor(),
])

transform_normalize = transforms.Normalize((0.5,), (0.5,))

occlusion = Occlusion(model)

sample = d_rsna[42]
transformed_img = transform(np.transpose(sample['img'], (1, 2, 0)))
input = transform_normalize(transformed_img)
input = input.unsqueeze(0)

attributions_occ = occlusion.attribute(input,
                                       strides=(1, 50, 50),
                                       target=int(sample['lab'][0]),
                                       sliding_window_shapes=(1, 60, 60),
                                       baselines=0,
                                       show_progress=True)

print(attributions_occ.squeeze().cpu().detach().numpy().shape)
print(transformed_img.squeeze().cpu().detach().numpy().shape)

_ = viz.visualize_image_attr_multiple(np.transpose(np.expand_dims(attributions_occ.squeeze().cpu().detach().numpy(), axis=0), (1, 2, 0)),
                                      np.transpose(np.expand_dims(transformed_img.squeeze().cpu().detach().numpy(), axis=0), (1, 2, 0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                      use_pyplot=True
                                      )

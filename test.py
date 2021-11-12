import numpy as np
import torch
import torchxrayvision as xrv
import viz as viz
from captum.attr import Occlusion

from data import load_rsna_dataset

d_rsna = load_rsna_dataset()
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')

# df, result = calculate_rsna_metrics(model, d_rsna, force=True)
#
# print(f'Accuracy: {result["accuracy"]}')
# print(f'Precision: {result["precision"]}')
# print(f'Recall: {result["recall"]}')
# print(f'F1: {result["f1"]}')

occlusion = Occlusion(model)

sample = d_rsna[42]

attributions_occ = occlusion.attribute(torch.from_numpy(sample["img"]).unsqueeze(0),
                                       strides=(1, 8, 8),
                                       target=int(sample['lab'][0]),
                                       sliding_window_shapes=(1, 15, 15),
                                       baselines=0)

_ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                      np.transpose(torch.from_numpy(sample["img"]).squeeze().cpu().detach().numpy(),
                                                   (1, 2, 0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                      )

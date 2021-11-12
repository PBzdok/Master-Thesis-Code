import matplotlib.pyplot as plt
import numpy as np
import torch
import torchxrayvision as xrv
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

i = 700

occlusion = Occlusion(model)

input1 = torch.from_numpy(np.expand_dims(d_rsna[i]['img'], axis=0)).float()
pred_label_idx = model(input1).argmax()

attr = occlusion.attribute(input1,
                           strides=(1, 30, 30),
                           target=pred_label_idx,
                           sliding_window_shapes=(1, 30, 30),
                           baselines=0
                           )

plt.imshow(input1[0, 0, :, :])
plt.contourf(attr[0, 0, :, :], alpha=0.6)
plt.colorbar()
plt.show()

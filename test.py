import matplotlib.pyplot as plt
import numpy as np
import torch
import torchxrayvision as xrv
from captum.attr import Occlusion

from data import load_rsna_dataset, calculate_rsna_metrics

d_rsna = load_rsna_dataset()
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')

df_predictions, metrics = calculate_rsna_metrics(model, d_rsna, force=False)

print(f'Accuracy: {metrics["accuracy"]}')
print(f'Precision: {metrics["precision"]}')
print(f'Recall: {metrics["recall"]}')
print(f'F1: {metrics["f1"]}')

occlusion = Occlusion(model)

for index, row in df_predictions.iterrows():
    patient_id = row['patientid']

    patient_index = d_rsna.csv[d_rsna.csv['patientid'] == patient_id].index.values[0]
    print(patient_index)

    model_input = d_rsna[patient_index]['img']
    model_input = np.expand_dims(model_input, axis=0)
    model_input = torch.from_numpy(model_input).float()

    pred_label_idx = model(model_input).argmax()

    attr = occlusion.attribute(model_input,
                               strides=(1, 30, 30),
                               target=pred_label_idx,
                               sliding_window_shapes=(1, 30, 30),
                               baselines=0,
                               show_progress=True
                               )

    plt.imshow(model_input[0, 0, :, :])
    plt.contourf(attr[0, 0, :, :], alpha=0.5)
    plt.colorbar()
    plt.savefig(f'./data/kaggle-pneumonia-jpg/occlusion/{patient_id}.jpg', bbox_inches='tight', dpi=150)
    plt.clf()

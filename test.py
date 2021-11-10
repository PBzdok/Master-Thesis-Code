import numpy as np
import sklearn.metrics
import torch
import torchvision
import torchxrayvision as xrv

from data import load_rsna_dataset

d_kag = load_rsna_dataset()

sample = d_kag[1]
print(dict(zip(d_kag.pathologies, sample["lab"])))

model = xrv.models.DenseNet(weights='densenet121-res224-rsna')
with torch.no_grad():
    out = model(torch.from_numpy(sample["img"]).unsqueeze(0)).cpu()

dict(zip(model.pathologies, zip(out[0].detach().numpy(), sample["lab"])))

print(d_kag.csv)

# outs = []
# labs = []
# with torch.no_grad():
#     for i in np.random.randint(0, len(d_nih), 100):
#         sample = d_nih[i]
#         labs.append(sample["lab"])
#         out = model(torch.from_numpy(sample["img"]).unsqueeze(0)).cpu()
#         out = torch.sigmoid(out)
#         outs.append(out.detach().numpy()[0])
#
# for i in range(14):
#     if len(np.unique(np.asarray(labs)[:, i])) > 1:
#         auc = sklearn.metrics.roc_auc_score(np.asarray(labs)[:, i], np.asarray(outs)[:, i])
#     else:
#         auc = "(Only one class observed)"
#     print(xrv.datasets.default_pathologies[i], auc)

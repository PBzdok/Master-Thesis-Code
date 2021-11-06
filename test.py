import numpy as np
import sklearn.metrics
import torch
import torchvision
import torchxrayvision as xrv

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
d_nih = xrv.datasets.NIH_Dataset(imgpath='./data/NIH/images-224',
                                 csvpath='./data/NIH/Data_Entry_2017.csv',
                                 transform=transform)
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d_nih)

sample = d_nih[40]
print(dict(zip(d_nih.pathologies, sample["lab"])))

model = xrv.models.DenseNet(weights="all")
with torch.no_grad():
    out = model(torch.from_numpy(sample["img"]).unsqueeze(0)).cpu()

dict(zip(model.pathologies, zip(out[0].detach().numpy(), sample["lab"])))

print(d_nih.csv)

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

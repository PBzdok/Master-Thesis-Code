import torchxrayvision as xrv

from data import load_rsna_dataset, calculate_rsna_metrics

d_rsna = load_rsna_dataset()
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')

accuracy, precision, recall, f1 = calculate_rsna_metrics(model, d_rsna)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1: {f1}')

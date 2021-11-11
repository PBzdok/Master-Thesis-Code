import torchxrayvision as xrv

from data import load_rsna_dataset, calculate_rsna_metrics

d_rsna = load_rsna_dataset()
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')

result = calculate_rsna_metrics(model, d_rsna)

print(f'Accuracy: {result["accuracy"]}')
print(f'Precision: {result["precision"]}')
print(f'Recall: {result["recall"]}')
print(f'F1: {result["f1"]}')

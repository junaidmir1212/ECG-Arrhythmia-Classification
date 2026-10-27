# evaluate.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import ECGArrhythmiaClassifier
from config import CLASS_NAMES

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    model = ECGArrhythmiaClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load('ecg_arrhythmia_best.pth', map_location=device))
    model.eval()
    
    inputs = torch.from_numpy(X_test).to(device)
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(inputs), 512):
            batch = inputs[i:i+512]
            all_preds.extend(torch.argmax(model(batch), dim=1).cpu().numpy())
            
    print("\n" + "=" * 50)
    print("AAMI EC57 BENCHMARK TEST (UNSEEN DS2 PATIENTS)")
    print("=" * 50)
    print(classification_report(y_test, all_preds, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, all_preds))

if __name__ == '__main__':
    evaluate()
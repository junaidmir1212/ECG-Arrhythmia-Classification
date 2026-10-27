# train.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from model import ECGArrhythmiaClassifier, FocalLoss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    counts = np.bincount(y_train, minlength=5)
    weights = len(y_train) / (5.0 * counts)
    alpha = torch.tensor(weights, dtype=torch.float32).to(device)
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=256, shuffle=False
    )
    
    model = ECGArrhythmiaClassifier(num_classes=5).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_f1 = 0.0
    for epoch in range(1, 16):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                preds.extend(torch.argmax(model(vx), dim=1).cpu().numpy())
                targets.extend(vy.numpy())
                
        macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
        print(f"Epoch {epoch:02d}/15 | DS2 Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), 'ecg_arrhythmia_best.pth')
            print(f"  -> Saved best model checkpoint (F1: {best_f1:.4f})")

if __name__ == '__main__':
    train()
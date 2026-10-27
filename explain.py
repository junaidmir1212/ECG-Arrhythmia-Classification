# explain.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ECGArrhythmiaClassifier
from config import CLASS_NAMES

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_acts)
        self.target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, m, inp, out): self.activations = out
    def _save_grads(self, m, gin, gout): self.gradients = gout[0]

    def generate(self, x, class_idx):
        self.model.eval()
        self.model.zero_grad()
        score = self.model(x)[0, class_idx]
        score.backward()
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.clamp(torch.sum(weights * self.activations, dim=1), min=0)
        cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        cam_up = torch.nn.functional.interpolate(cam.unsqueeze(1), size=x.shape[2], mode='linear', align_corners=False)
        return cam_up.squeeze().detach().cpu().numpy()

def explain():
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    model = ECGArrhythmiaClassifier(num_classes=5)
    model.load_state_dict(torch.load('ecg_arrhythmia_best.pth', map_location='cpu'))
    
    cam_engine = GradCAM1D(model, target_layer=model.feature_extractor[8])
    idx = np.where(y_test == 2)[0][0]
    beat = X_test[idx, 0]
    cam = cam_engine.generate(torch.from_numpy(X_test[idx:idx+1]), class_idx=2)
    
    time_ms = np.linspace(-355, 355, len(beat))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(time_ms, beat, color='black', lw=1.5)
    sc = ax1.scatter(time_ms, beat, c=cam, cmap='inferno', s=14, zorder=3)
    ax1.set_title(f"1D Grad-CAM Attribution | Class: {CLASS_NAMES[2]}")
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.colorbar(sc, ax=ax1, label='Diagnostic Importance')
    
    ax2.imshow(cam[np.newaxis, :], aspect='auto', cmap='inferno', extent=[time_ms[0], time_ms[-1], 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel("Time from R-Peak (ms)")
    
    plt.tight_layout()
    plt.savefig('gradcam_output.png', dpi=300)
    print("Heatmap saved to 'gradcam_output.png'.")

if __name__ == '__main__':
    explain()
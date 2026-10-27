# advanced_xai.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import ECGArrhythmiaClassifier

def integrated_gradients(model, input_tensor, target_class, steps=50):
    """
    Computes Integrated Gradients against a zero-baseline.
    """
    model.eval()
    baseline = torch.zeros_like(input_tensor)
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)]
    
    grads = []
    for step_input in scaled_inputs:
        step_input = step_input.clone().detach().requires_grad_(True)
        logits = model(step_input)
        score = logits[0, target_class]
        score.backward()
        grads.append(step_input.grad.detach())
        
    avg_grads = torch.mean(torch.stack(grads), dim=0)
    ig_attributions = (input_tensor - baseline) * avg_grads
    return ig_attributions.squeeze().cpu().numpy()

def generate_counterfactual(model, input_tensor, target_class=0, lr=0.01, max_iter=200, lambda_dist=0.5):
    """
    Generates a counterfactual ECG beat by finding the minimal perturbation delta
    that causes the model to classify an abnormal beat as Normal (class 0).
    """
    model.eval()
    delta = torch.zeros_like(input_tensor, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        perturbed = input_tensor + delta
        logits = model(perturbed)
        
        # Loss: minimize CrossEntropy to Target Class + L2 distance of delta
        loss_ce = F.cross_entropy(logits, torch.tensor([target_class], device=input_tensor.device))
        loss_dist = torch.norm(delta, p=2)
        total_loss = loss_ce + lambda_dist * loss_dist
        
        total_loss.backward()
        optimizer.step()
        
        # Early break if target confidence exceeds 90%
        prob = F.softmax(logits, dim=1)[0, target_class].item()
        if prob > 0.90:
            break

    counterfactual_beat = (input_tensor + delta).squeeze().detach().cpu().numpy()
    delta_array = delta.squeeze().detach().cpu().numpy()
    return counterfactual_beat, delta_array

def run_advanced_xai():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGArrhythmiaClassifier(num_classes=5).to(device)
    model.load_state_dict(torch.load('ecg_arrhythmia_best.pth', map_location=device))
    
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Select a Ventricular Ectopic beat (Class 2)
    v_idx = np.where(y_test == 2)[0][0]
    beat_signal = X_test[v_idx, 0]
    input_t = torch.from_numpy(X_test[v_idx:v_idx+1]).to(device)

    # 1. Compute Integrated Gradients
    ig_attr = integrated_gradients(model, input_t, target_class=2)

    # 2. Compute Counterfactual beat (what would make it Class 0: Normal?)
    cf_beat, delta_perturbation = generate_counterfactual(model, input_t, target_class=0)

    # 3. Plot side-by-side comparison
    time_ms = np.linspace(-355, 355, len(beat_signal))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Integrated Gradients
    axes[0].plot(time_ms, beat_signal, color='black', label='Abnormal Beat (Class V)')
    axes[0].fill_between(time_ms, beat_signal, beat_signal + ig_attr * 10, color='red', alpha=0.5, label='IG Attribution')
    axes[0].set_title("1. Point-wise Integrated Gradients (Attribution)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # Plot 2: Counterfactual Morphology
    axes[1].plot(time_ms, beat_signal, 'k--', alpha=0.6, label='Original V Beat')
    axes[1].plot(time_ms, cf_beat, color='green', lw=1.8, label='Counterfactual Beat (Class N)')
    axes[1].set_title("2. Counterfactual Minimal Repair (Target: Normal)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    # Plot 3: The Delta Perturbation
    axes[2].plot(time_ms, delta_perturbation, color='blue', label='Required Morphological Shift (Δ)')
    axes[2].set_title("3. Morphological Perturbation Delta (What the model required to flip decision)")
    axes[2].set_xlabel("Time relative to R-Peak (ms)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('advanced_xai_counterfactual.png', dpi=300)
    print("Advanced XAI explanation saved to 'advanced_xai_counterfactual.png'.")

if __name__ == '__main__':
    run_advanced_xai()
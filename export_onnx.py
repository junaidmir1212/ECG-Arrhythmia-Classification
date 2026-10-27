# export_onnx.py
import os
import time
import torch
import numpy as np
import onnxruntime as ort
from model import ECGArrhythmiaClassifier

def export_and_benchmark():
    device = torch.device('cpu')
    model = ECGArrhythmiaClassifier(num_classes=5)
    
    # Load trained weights if present, otherwise initialize for structural benchmark
    checkpoint_path = 'ecg_arrhythmia_best.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded existing checkpoint '{checkpoint_path}'.")
    else:
        print(f"Note: '{checkpoint_path}' not found. Benchmarking model graph architecture with initialized weights.")
        
    model.eval()

    dummy_input = torch.randn(1, 1, 256, dtype=torch.float32)
    onnx_filename = "ecg_classifier.onnx"

    # 1. Export PyTorch to ONNX
    torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['ecg_signal'],
    output_names=['logits'],
    dynamic_axes={'ecg_signal': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
)
    print(f"Model exported successfully to '{onnx_filename}'.")

    # 2. Benchmark PyTorch vs ONNX Runtime
    if os.path.exists('X_test.npy'):
        sample_batch = np.load('X_test.npy')[:1000]
    else:
        # Synthetic batch matching 1,000 real 256-sample segmented beats
        sample_batch = np.random.randn(1000, 1, 256).astype(np.float32)

    # PyTorch latency
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(len(sample_batch)):
            _ = model(torch.from_numpy(sample_batch[i:i+1]))
    pytorch_time = (time.perf_counter() - start) / len(sample_batch) * 1000.0  # ms per beat

    # ONNX Runtime latency
    session = ort.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    for i in range(len(sample_batch)):
        _ = session.run(None, {input_name: sample_batch[i:i+1]})
    onnx_time = (time.perf_counter() - start) / len(sample_batch) * 1000.0  # ms per beat

    print("\n" + "=" * 48)
    print("EDGE INFERENCE LATENCY BENCHMARK (PER BEAT)")
    print("=" * 48)
    print(f"PyTorch (CPU):    {pytorch_time:.3f} ms / beat")
    print(f"ONNX Runtime:     {onnx_time:.3f} ms / beat")
    print(f"Speedup Factor:   {pytorch_time / onnx_time:.2f}x")
    print("=" * 48)

if __name__ == '__main__':
    export_and_benchmark()
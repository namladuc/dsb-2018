import os
import subprocess

models = [
    'FusionUnet2D-test01_resnet18',
    'FusionUnet2D-test02_resnet152',
    'FusionUnet2D-test03_timmb3',
    'FusionUnet2DPP-test02_timmb3_deeply_baseline',
    'Unet2D-Baseline_baseline'
]

for m in models:
    csv_path = f"data/model_checkpoint/submission_{m}.csv"
    output_dir = f"vis_{m}"
    cmd = ["conda", "run", "-n", "cta", "python", "visualize_inference.py", csv_path, output_dir]
    print(f"Visualizing {m}...")
    subprocess.run(cmd)

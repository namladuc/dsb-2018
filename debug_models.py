import os
import subprocess
import shlex
import torch

checkpoint_dir = "data/model_checkpoint"
output_dir = "debug_inference_results"
os.makedirs(output_dir, exist_ok=True)

models = os.listdir(checkpoint_dir)
device = "cpu" # Running on CPU for debug stability

for model_name in models:
    model_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.isdir(model_path):
        continue
    
    arg_file = os.path.join(model_path, "arg.md")
    if not os.path.exists(arg_file):
        print(f"Skipping {model_name}: arg.md not found")
        continue
        
    checkpoint_file = None
    for f in os.listdir(model_path):
        if f.endswith(".bin") or f.endswith(".pth"):
            checkpoint_file = os.path.join(model_path, f)
            break
            
    if not checkpoint_file:
        print(f"Skipping {model_name}: no checkpoint file found")
        continue
        
    with open(arg_file, "r") as f:
        extra_args = f.read().strip()
        
    cmd = [
        "python", "inference.py",
        "--showcase",
        "--debug_vis",
        "--input_path", "data"
    ]
    
    cmd.extend(shlex.split(extra_args))
    
    submission_path = os.path.join(output_dir, f"submission_{model_name}.csv")
    
    cmd.extend([
        "--output_path", submission_path,
        "--using_wandb", "0",
        "--checkpoint_path", checkpoint_file,
        "--numWorker", "0",
        "--device", device
    ])
    
    print(f"\n" + "="*50)
    print(f"DEBUG INFERENCE for {model_name}...")
    
    try:
        # We need to capture and move the debug images because they all save to 'debug_inference/'
        subprocess.run(cmd, check=True)
        
        # Move images to a model-specific folder
        model_debug_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_debug_dir, exist_ok=True)
        
        if os.path.exists("debug_inference"):
            for f in os.listdir("debug_inference"):
                os.rename(os.path.join("debug_inference", f), os.path.join(model_debug_dir, f))
            os.rmdir("debug_inference")
            
        print(f"Successfully verified {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for {model_name}: {e}")

import os
import subprocess
import shlex
import torch

checkpoint_dir = "/kaggle/input/datasets/namsiunhon/dsb2018-ckpt/model_checkpoint"
models = os.listdir(checkpoint_dir)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

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
        
    # Build command
    # Just use 'python' because we will run this script itself with 'conda run -n cta'
    cmd = [
        "python", "inference.py"
    ]
    
    # Prepend extra args from arg.md
    cmd.extend(shlex.split(extra_args))
    
    # Target submission path
    submission_path = os.path.join(model_path, "submission.csv")
    
    # Ensure redundant/conflicting args are handled by appending our overrides at the end
    cmd.extend([
        "--input_path", "/kaggle/working/dsb-data-2018",
        "--output_path", submission_path,
        "--using_wandb", "0",
        "--checkpoint_path", checkpoint_file,
        "--fold_selected", "0",
        "--numWorker", "0",
        "--device", device
    ])
    
    print(f"\n" + "="*50)
    print(f"Running inference for {model_name}...")
    # print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated {submission_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running inference for {model_name}: {e}")

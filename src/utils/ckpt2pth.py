import torch
import argparse
from pathlib import Path

def convert_ckpt_to_pth(ckpt_path, output_path='pretrained'):
    """
    Convert a PyTorch checkpoint file to a .pth file
    Args:
        ckpt_path (str): Path to the checkpoint file
        output_path (str): Path where to save the .pth file
    """
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    
    # If checkpoint is a state dict, use it directly
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Remove 'module.' prefix if it exists
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    output_path = Path(output_path) / Path(ckpt_path).name.replace('.ckpt', '.pth')
    
    # Save as .pth file
    torch.save(state_dict, output_path)
    print(f"Converted {ckpt_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert .ckpt to .pth')
    parser.add_argument('ckpt_path', type=str, help='Path to checkpoint file')
    parser.add_argument('--output', type=str, help='Output path for .pth file', default='pretrained')
    
    args = parser.parse_args()
    convert_ckpt_to_pth(args.ckpt_path, args.output)

if __name__ == '__main__':
    main()
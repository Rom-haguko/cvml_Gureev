import torch
import matplotlib.pyplot as plt
from train import Encoder, Decoder, ImageDataset

def get_inference_pair(m_idx, dev):
    e_net = Encoder().to(dev)
    d_net = Decoder().to(dev)
    
    e_net.load_state_dict(torch.load(f"encoder_{m_idx}.pth", map_location=dev))
    d_net.load_state_dict(torch.load(f"decoder_{m_idx}.pth", map_location=dev))
    
    e_net.eval()
    d_net.eval()
    
    ds = ImageDataset(n=1, size=256, mode=m_idx)
    img_tensor, _ = ds[0]
    
    with torch.inference_mode():
        x = img_tensor.unsqueeze(0).to(dev)
        reconstruction = d_net(e_net(x))
        
    original = img_tensor[0].cpu().numpy()
    restored = reconstruction[0][0].cpu().numpy()
    
    return original, restored

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    fig, grid = plt.subplots(4, 2, figsize=(10, 14))
    fig.suptitle("Оригинал vs Восстановление", fontsize=16)
    
    for row, mode_num in enumerate([1, 2, 3, 4]):
        src, res = get_inference_pair(mode_num, device)
        
        grid[row, 0].imshow(src, cmap='gray')
        grid[row, 0].set_title(f"Режим {mode_num} (Оригинал)")
        grid[row, 0].axis('off')
        
        grid[row, 1].imshow(res, cmap='gray')
        grid[row, 1].set_title(f"Режим {mode_num} (Восстановлено)")
        grid[row, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("output_comparison.png", dpi=200)
    plt.show()
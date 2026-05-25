import unet_road
from pathlib import Path
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    path = Path(__file__).resolve().parent / "roads"
    
    ds = unet_road.RoadsDataset(path)

    model = unet_road.UNet().to(device)
    model.load_state_dict(torch.load('unet_road.pth'))
    model.eval() 
    
    img, mask = ds[0]
    img, mask = img.to(device), mask.to(device)

    with torch.no_grad():
        result = model(img.unsqueeze(0))

    mask = mask.detach().squeeze().cpu().numpy()
    result = result.detach().squeeze().cpu().numpy()
    
    result[result > 0] = 1
    result[result <= 0] = 0
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(mask)
    plt.title('orig')
    
    plt.subplot(132)
    plt.imshow(result)
    plt.title('result')
    
    plt.subplot(133)
    plt.imshow(mask - result)
    plt.title('diff')
    
    plt.savefig('compare.png')
    plt.show()
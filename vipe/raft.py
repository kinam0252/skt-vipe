from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.io import read_video, write_jpeg
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model & weights
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
model = raft_large(weights=weights, progress=False).to(device).eval()

# Load frames
video_path = "/home/nas4_user/hoiyeongjin/repos/Project/SKT_1027/vipe/2125_2_cut1/rgb/2125_2_cut1.mp4"
frames, _, _ = read_video(video_path, output_format="TCHW")

def preprocess(img1, img2):
    img1 = F.resize(img1, size=[520, 960], antialias=False)
    img2 = F.resize(img2, size=[520, 960], antialias=False)
    img1, img2 = transforms(img1.unsqueeze(0), img2.unsqueeze(0))
    return img1[0], img2[0]

output_folder = "/home/nas4_user/hoiyeongjin/repos/Project/SKT_1027/vipe/vipe/tmp/"
for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
    img1, img2 = preprocess(img1, img2)
    with torch.no_grad():
        list_of_flows = model(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
    predicted_flow = list_of_flows[-1][0].cpu()

    # Clamp for visualization stability
    predicted_flow = predicted_flow.clamp(-20, 20) / 20.0

    flow_img = flow_to_image(predicted_flow)
    write_jpeg(flow_img, output_folder + f"predicted_flow_{i:03d}.jpg")


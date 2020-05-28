import argparse
import torchvision
import torch
from PIL import Image


def test(data, network):
    output, _ = network(data)
    return output


parser = argparse.ArgumentParser(
    description = 'Testing utility for paper "Investigation on different loss function in image autoencoder"')
parser.add_argument('target', type = str, help = "Input image")
parser.add_argument('output', type = str, help = "Output image")
parser.add_argument('--model_name', type = str, dest = "model_name", help = "Model name for saving & restoring model",
                    default = "saved.pkl")
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
filename = args.target
output = args.output
model_name = args.model_name
network = torch.load(model_name)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    network.cuda()
else:
    args.device = torch.device('cpu')
data = Image.open(filename)
data = torchvision.transforms.ToTensor()(data).unsqueeze_(0).to(args.device)
result = test(data, network)
result = torchvision.transforms.ToPILImage()(result.squeeze_(0).to("cpu"))
result.save(output)

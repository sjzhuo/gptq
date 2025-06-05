import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm

from gptq import GPTQ
from quant import Quantizer
from modelutils import find_layers


def get_calibration_loader(path, nsamples, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    it = iter(loader)
    imgs = []
    for _ in range(nsamples):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(loader)
            x, _ = next(it)
        imgs.append(x)
    imgs = torch.cat(imgs, dim=0)
    return imgs


@torch.no_grad()
def quantize_resnet(model, calib_data, dev, wbits=4, groupsize=-1, percdamp=0.01):
    model.eval()
    model.to(dev)
    layers = find_layers(model)
    quantizers = {}

    def add_batch(name, gptq):
        def hook(module, inp, out):
            gptq.add_batch(inp[0].data, out.data)
        return hook

    for name, layer in layers.items():
        gptq = GPTQ(layer)
        gptq.quantizer = Quantizer()
        gptq.quantizer.configure(wbits, perchannel=True, sym=True)
        handle = layer.register_forward_hook(add_batch(name, gptq))
        _ = model(calib_data.to(dev))
        handle.remove()
        gptq.fasterquant(percdamp=percdamp, groupsize=groupsize)
        quantizers[name] = gptq.quantizer
        gptq.free()

    return model, quantizers


@torch.no_grad()
def evaluate(model, dataloader, dev):
    model.eval()
    model.to(dev)
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(dev)
        y = y.to(dev)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="GPTQ quantization for ResNet18")
    parser.add_argument("data", type=str, help="Path to calibration dataset")
    parser.add_argument("--wbits", type=int, default=4, choices=[2,3,4,8,16])
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("resnet18", pretrained=True)

    calib_data = get_calibration_loader(args.data, args.nsamples)
    model, quantizers = quantize_resnet(model, calib_data, dev, args.wbits, args.groupsize, args.percdamp)

    if args.save:
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()

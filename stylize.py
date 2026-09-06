import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils.models import Decoder, VGGEncoder, load_decoder_weights
from utils.utils import adaptive_instance_normalization


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_models(vgg_path, decoder_path, device=None):
    device = device or get_device()
    encoder = VGGEncoder(vgg_path).to(device).eval()
    decoder = Decoder().to(device)
    load_decoder_weights(decoder, decoder_path, device)
    return encoder, decoder.eval(), device


def image_tensor(image, max_size):
    width, height = image.size
    scale = min(1, max_size / max(width, height))
    width, height = max(8, int(width * scale) // 8 * 8), max(8, int(height * scale) // 8 * 8)
    return transforms.ToTensor()(image.resize((width, height), Image.Resampling.LANCZOS)).unsqueeze(0)


def transfer(content_image, style_image, encoder, decoder, device, alpha=1.0, max_size=512):
    content = image_tensor(content_image.convert('RGB'), max_size).to(device)
    style = image_tensor(style_image.convert('RGB'), max_size).to(device)
    with torch.no_grad():
        content_features = encoder(content, is_test=True)
        style_features = encoder(style, is_test=True)
        features = adaptive_instance_normalization(content_features, style_features)
        output = decoder(alpha * features + (1 - alpha) * content_features)
    return transforms.ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))


def main():
    parser = argparse.ArgumentParser(description='Generate one AdaIN style-transfer image.')
    parser.add_argument('--content', type=Path, required=True)
    parser.add_argument('--style', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/stylized.png'))
    parser.add_argument('--vgg', type=Path, default=Path('models/vgg_normalised.pth'))
    parser.add_argument('--decoder', type=Path, default=Path('artifacts/decoder.pth'))
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max-size', type=int, default=512)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1.')
    for path in (args.content, args.style, args.vgg, args.decoder):
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    encoder, decoder, device = load_models(args.vgg, args.decoder)
    transfer(Image.open(args.content), Image.open(args.style), encoder, decoder, device, args.alpha, args.max_size).save(args.output, quality=95)
    print(f'Saved stylized image: {args.output}')


if __name__ == '__main__':
    main()

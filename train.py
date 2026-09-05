import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from utils.models import Decoder, VGGEncoder, load_decoder_weights
from utils.utils import ImageFolderDataset, adaptive_instance_normalization, calc_mean_std, get_transform


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compact AdaIN decoder fine-tuning.')
    parser.add_argument('--content-dir', type=Path,
                        default=Path('content_data'))
    parser.add_argument('--style-dir', type=Path, default=Path('style_data'))
    parser.add_argument('--vgg', type=Path,
                        default=Path('models/vgg_normalised.pth'))
    parser.add_argument('--decoder', type=Path, default=Path(
        'models/decoder.pth'), help='Optional decoder to fine-tune')
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/decoder.pth'))
    parser.add_argument('--sample-output', type=Path,
                        default=Path('outputs/training_sample.png'))
    parser.add_argument(
        '--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--final-size', type=int, default=128)
    parser.add_argument('--source-size', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=8,
                        help='0 processes every batch')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--content-weight', type=float, default=1.0)
    parser.add_argument('--style-weight', type=float, default=5.0)
    return parser.parse_args()


def get_device(choice):
    if choice != 'auto':
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    args = parse_arguments()
    device = get_device(args.device)
    if not args.vgg.is_file():
        raise FileNotFoundError(f'Missing encoder weights: {args.vgg}')

    content_dataset = ImageFolderDataset(
        args.content_dir, get_transform(args.source_size, True, args.final_size))
    style_dataset = ImageFolderDataset(
        args.style_dir, get_transform(args.source_size, True, args.final_size))
    if not content_dataset or not style_dataset:
        raise ValueError(
            'Content and style directories must both contain at least one image.')

    loader_args = dict(batch_size=args.batch_size, shuffle=True,
                       pin_memory=device.type == 'cuda', drop_last=False)
    content_loader = DataLoader(content_dataset, **loader_args)
    style_loader = DataLoader(style_dataset, **loader_args)
    print(
        f'Content batches: {len(content_loader)}; style batches: {len(style_loader)}')

    encoder = VGGEncoder(args.vgg).to(device).eval()
    decoder = Decoder().to(device)
    if args.decoder.is_file():
        load_decoder_weights(decoder, args.decoder, device)
        print(f'Fine-tuning decoder: {args.decoder}')
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss()

    print(f'Training on {device}...')
    for epoch in range(args.epochs):
        progress = tqdm(zip(content_loader, style_loader),
                        total=min(len(content_loader), len(style_loader)))
        losses = []
        for step, (content, style) in enumerate(progress, start=1):
            content, style = content.to(device), style.to(device)
            with torch.no_grad():
                content_features = encoder(content)
                style_features = encoder(style)
                target = adaptive_instance_normalization(
                    content_features[-1], style_features[-1])

            generated = decoder(target)
            generated_features = encoder(generated)
            content_loss = mse_loss(
                generated_features[-1], target) * args.content_weight
            style_loss = sum(
                mse_loss(calc_mean_std(generated_feature)[0], calc_mean_std(style_feature)[0]) +
                mse_loss(calc_mean_std(generated_feature)[
                         1], calc_mean_std(style_feature)[1])
                for generated_feature, style_feature in zip(generated_features, style_features)
            ) * args.style_weight
            loss = content_loss + style_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.set_description(f'loss={loss.item():.4f}')
            if args.max_steps and step >= args.max_steps:
                break
        print(
            f'Epoch {epoch + 1} average loss: {sum(losses) / len(losses):.4f}')

    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.sample_output.parent.mkdir(exist_ok=True, parents=True)
    torch.save(decoder.state_dict(), args.output)
    with torch.no_grad():
        save_image(torch.cat([content, style, generated],
                   dim=0), args.sample_output, nrow=args.batch_size)
    print(f'Saved checkpoint: {args.output}')
    print(f'Saved preview: {args.sample_output}')


if __name__ == '__main__':
    main()

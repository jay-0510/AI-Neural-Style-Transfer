# Neural Style Transfer

A local, browser-based neural style transfer project built with PyTorch, Adaptive Instance Normalization (AdaIN), and Flask. It takes a content image and a style-reference image, then generates a new image that keeps the content structure while adopting the reference image's visual style.

The project is designed to run locally on macOS with Apple Silicon acceleration when available. No cloud API, account, or internet connection is needed after the one-time model-weight download.

## Features

- Transfer the look of a painting, sketch, watercolor, abstract work, or other reference image to a photo.
- Upload any local PNG, JPG, or JPEG file through a browser. Uploaded images do not need to be in the training folders.
- Control the blend between content and style with a style-strength value from `0.0` to `1.0`.
- Use Apple Metal Performance Shaders (MPS) automatically on compatible Macs, with CUDA and CPU fallbacks.
- Generate images from the command line without opening the web interface.
- Fine-tune the decoder with a compact, low-storage configuration.
- Save only one final decoder checkpoint, rather than optimizer states and many per-epoch checkpoints.

## How It Works

This implementation uses the AdaIN method.

1. A fixed, pretrained VGG-19 encoder extracts visual features from the content and style images.
2. Adaptive Instance Normalization changes the content feature statistics to match the style feature statistics.
3. A trained decoder turns the combined features back into an RGB image.
4. The `alpha` value blends the transformed features with the original content features.

The encoder remains frozen. Training fine-tunes only the decoder, which reduces the compute and storage required compared with training a full model.

## Example Output

The project includes a generated example:

```text
outputs/brad_pitt_sketch.png
```

It uses `content_data/brad_pitt.jpg` as content and `style_data/sketch.png` as the style reference.

## Requirements

- macOS, Linux, or Windows
- Python 3.11 recommended
- Conda recommended for environment management
- At least 1 GB of free disk space for Python packages, model weights, and generated files
- Apple Silicon Macs can use MPS acceleration automatically

The current configured environment is named `nst_env` and uses Python 3.11 with PyTorch 2.2.2.

## Project Structure

```text
Neural_Style_Transfer/
├── app.py                     # Flask web application
├── train.py                   # Compact decoder fine-tuning script
├── stylize.py                 # Command-line image generation script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── content_data/              # Small content-image training dataset
├── style_data/                # Small style-image training dataset
├── models/                    # Downloaded VGG encoder and initial decoder weights
├── artifacts/                 # Final trained decoder checkpoint
├── outputs/                   # Generated command-line and training-preview images
├── static/uploads/            # Images uploaded through the browser and web results
├── templates/index.html       # Web interface template
├── utils/models.py            # VGG encoder and decoder architectures
├── utils/utils.py             # Dataset, image transforms, and AdaIN functions
├── examples/                  # Images shown in the web interface
└── research_docs/             # Reference papers and AdaIN diagram
```

`models/`, `artifacts/`, and `outputs/` are intentionally ignored by Git because they contain large downloaded or generated files.

## Installation From Scratch

Run these commands from the `Neural_Style_Transfer` folder.

### 1. Create the Conda environment

```bash
conda create -n nst_env python=3.11 -y
conda run -n nst_env pip install -r requirements.txt
```

If the environment already exists, verify it with:

```bash
conda run -n nst_env python --version
```

### 2. Download the required AdaIN weights

Create the model directory and download the two official weights:

```bash
mkdir -p models
curl --fail --location --output models/vgg_normalised.pth https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
curl --fail --location --output models/decoder.pth https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth
```

Required files:

| File                        | Purpose                          | Approximate size |
| --------------------------- | -------------------------------- | ---------------: |
| `models/vgg_normalised.pth` | Frozen VGG-19 feature encoder    |            80 MB |
| `models/decoder.pth`        | Initial pretrained AdaIN decoder |            14 MB |

The initial decoder is used as a starting point. The project fine-tunes it using the included image folders.

### 3. Confirm the setup

```bash
conda run -n nst_env python -m py_compile app.py train.py stylize.py utils/models.py utils/utils.py
```

## Low-Storage Training

Run the compact default training job:

```bash
conda run -n nst_env python train.py
```

Default training configuration:

| Setting                   |  Value | Why it saves resources              |
| ------------------------- | -----: | ----------------------------------- |
| Crop size                 | 128 px | Low GPU/CPU memory usage            |
| Source resize             | 160 px | Small image preprocessing cost      |
| Batch size                |      1 | Lowest practical memory use         |
| Epochs                    |      1 | Short local fine-tuning run         |
| Maximum steps             |      8 | Bounded compute time                |
| Optimizer checkpoints     |      0 | Avoids large duplicate files        |
| Final decoder checkpoints |      1 | Stores only `artifacts/decoder.pth` |

Training creates:

```text
artifacts/decoder.pth          # Fine-tuned decoder, about 13 MB
outputs/training_sample.png    # Content, style, and generated training preview
```

To train for more steps while keeping the same low image size:

```bash
conda run -n nst_env python train.py --epochs 2 --max-steps 16
```

To process every available batch in each epoch, use `--max-steps 0`:

```bash
conda run -n nst_env python train.py --epochs 1 --max-steps 0
```

The included datasets are intentionally small, so this is a demonstration fine-tuning workflow rather than a replacement for large-scale AdaIN training. For better generalization, use a large and legally usable content-image dataset, but expect longer training and more disk use.

## Use the Web Application

Start the Flask server:

```bash
conda run -n nst_env python app.py
```

Open this URL in Chrome, Safari, or another browser:

```text
http://localhost:5001
```

Do not use port `5000` on this Mac. macOS Control Center already occupies that port and returns a `403` response. The application is configured for port `5001`.

### Browser Workflow

1. Select a content image from anywhere on your Mac.
2. Select a style image from anywhere on your Mac.
3. Set Style Strength between `0.0` and `1.0`.
4. Select `Transfer Style`.
5. View the generated result and select `Download Result` if desired.

Content and style images can be downloaded from the internet first, then uploaded from your computer. They do not need to be copied into `content_data/` or `style_data/`.

Use these values as a starting point:

| Style strength | Expected result                   |
| -------------: | --------------------------------- |
|          `0.0` | Nearly the original content image |
| `0.3` to `0.6` | Balanced content and style        |
| `0.7` to `1.0` | Strong style transfer             |

Uploaded files and browser-generated results are stored locally in `static/uploads/`. Remove unwanted files from that directory occasionally if disk space is limited.

## Generate an Image From the Command Line

The command-line tool is useful for repeatable experiments and does not require the web server.

```bash
conda run -n nst_env python stylize.py --content content_data/brad_pitt.jpg --style style_data/sketch.png --output outputs/brad_pitt_sketch.png
```

Use your own images by replacing the input paths:

```bash
conda run -n nst_env python stylize.py --content "/path/to/my_photo.jpg" --style "/path/to/style_reference.jpg" --output outputs/my_stylized_image.png --alpha 0.8 --max-size 512
```

Command-line options:

| Option       | Default                     | Description                                               |
| ------------ | --------------------------- | --------------------------------------------------------- |
| `--content`  | Required                    | Content image path                                        |
| `--style`    | Required                    | Style-reference image path                                |
| `--output`   | `outputs/stylized.png`      | Generated image path                                      |
| `--alpha`    | `1.0`                       | Style strength, from `0` to `1`                           |
| `--max-size` | `512`                       | Maximum long edge in pixels; lower values use less memory |
| `--vgg`      | `models/vgg_normalised.pth` | Encoder weights path                                      |
| `--decoder`  | `artifacts/decoder.pth`     | Fine-tuned decoder checkpoint path                        |

The image is resized so its dimensions are multiples of 8, which is required by the decoder upsampling architecture.

## Storage Management

The current compact project artifacts use approximately:

| Location                          | Typical size | Keep?                                        |
| --------------------------------- | -----------: | -------------------------------------------- |
| `models/`                         |        95 MB | Yes, needed for training and inference       |
| `artifacts/decoder.pth`           |        13 MB | Yes, needed by the web app and CLI           |
| `outputs/`                        |       Varies | Keep only desired results                    |
| `static/uploads/`                 |       Varies | Delete old uploads and results when finished |
| `content_data/` and `style_data/` |         7 MB | Keep for compact training                    |

Avoid increasing image resolution, batch size, epochs, and checkpoint frequency at the same time on a low-storage machine.

## Troubleshooting

### `localhost:5001` does not open

Keep the Terminal window that started `app.py` open. A working startup message includes `Running on http://127.0.0.1:5001`.

Check whether another program occupies the port:

```bash
lsof -nP -iTCP:5001 -sTCP:LISTEN
```

### `ModuleNotFoundError: No module named 'torch'`

The wrong Python environment is being used. Run commands through Conda:

```bash
conda run -n nst_env python app.py
```

### `Missing file: models/...` or missing encoder weights

Run the model-download commands in the Installation section, then verify:

```bash
ls -lh models
```

### The web app says the trained decoder is missing

Run compact training once:

```bash
conda run -n nst_env python train.py
```

This creates `artifacts/decoder.pth`, which the web app requires.

### The generated image is too slow or uses too much memory

Use a smaller command-line output size:

```bash
conda run -n nst_env python stylize.py --content content_data/brad_pitt.jpg --style style_data/sketch.png --output outputs/small_result.png --max-size 256
```

For training, retain the default `--final-size 128` and `--batch-size 1`.

### MPS is unavailable

The application automatically falls back to CUDA if available, otherwise CPU. CPU execution is supported but will be slower.

## Important Notes

- Use images you own, have permission to use, or that are licensed for your intended purpose.
- Style transfer may preserve some colors, lighting, and structure from the content image; results vary by image pair.
- Clear, well-lit content photos and visually distinct style images generally give better results.
- This is a local development application. Before publishing it publicly, configure a strong `FLASK_SECRET_KEY`, restrict file sizes, validate uploaded image contents, and run behind a production WSGI server.

## References

- Xun Huang and Serge Belongie, _Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization_.
- Official AdaIN PyTorch implementation and pretrained weights: https://github.com/naoto0804/pytorch-AdaIN
- Project papers and a method diagram are available in `research_docs/`.

# Assignment Three - Image Colorization

## 1) Convolutional Neural Network for Image Colorization

Dataset used: CIFAR-10 (horse category only), with `32x32` RGB images and grayscale inputs.

## 2) Colourization as Regression

### 2.1 Convolution layers in regression CNN

The regression model (`RegressionCNN`) contains **6 convolution layers**:

1. `Conv2d(1 -> num_filters, k=3)` + BN + ReLU + MaxPool
2. `Conv2d(num_filters -> 2*num_filters, k=3)` + BN + ReLU + MaxPool
3. `Conv2d(2*num_filters -> 2*num_filters, k=3)` + BN + ReLU
4. `Conv2d(2*num_filters -> num_filters, k=3)` + BN + ReLU + Upsample
5. `Conv2d(num_filters -> 3, k=3)` + BN + ReLU + Upsample
6. `MyConv2d(3 -> 3, k=3)` (final RGB prediction)

With `num_filters=32`, channel progression is:
`1 -> 32 -> 64 -> 64 -> 32 -> 3 -> 3`.

### 2.2 Training epochs used

Default configuration in notebook trains for:
- **25 epochs** (`'epochs': 25`)

### 2.3 Training variations (required discussion)

Recommended variation runs (same hyperparameters otherwise):
- Short: 5 epochs
- Medium: 15 epochs
- Baseline: 25 epochs
- Extended: 40 epochs

Executed baseline run result (`epochs=25`):
- Final train loss: **0.0088**
- Final validation loss: **0.0087**

Observed trend in the executed run:
- Loss drops quickly in early epochs and then improves gradually.
- The model converges stably by the final epochs without divergence.

## 3) Colourization as Classification

### 3.1 Quantized color setup

- 24 representative colors are used (k-means centers from `colour_kmeans24_cat7.npy`).
- Task is pixel-wise classification into one of 24 classes.

### 3.2 Classification CNN architecture

Implemented `CNN` in `colourization.ipynb` with required components:
- Uses custom `MyConv2d` layers (instead of `nn.Conv2d`)
- Uses `nn.ReLU`, `nn.BatchNorm2d`, `nn.MaxPool2d`
- Output layer predicts `num_colours` channels (24 logits per pixel)

Convolution stack (6 conv layers):
1. `MyConv2d(in -> 32)`
2. `MyConv2d(32 -> 64)`
3. `MyConv2d(64 -> 64)`
4. `MyConv2d(64 -> 32)`
5. `MyConv2d(32 -> 32)`
6. `MyConv2d(32 -> 24)` (final class logits)

### 3.3 Comparison against regression model

Executed classification CNN result (`epochs=25`):
- Final train loss: **1.5479**
- Final validation loss: **1.5716**
- Final validation accuracy: **41.7%**

Qualitative comparison from generated images (`outputs/colourization_cnn` vs `outputs/regression_cnn`):
- Classification outputs are generally more vivid and discrete in color assignment.
- Regression outputs are smoother and can look less saturated.
- Both methods preserve coarse horse structure; classification better matches the predefined palette.

## 4) Skip Connections (UNet)

### 4.1 Skip-connection design

Implemented `UNet` with explicit skip links:
- Decoder receives encoder features:
  - Mid decoder output concatenated with first encoder block output
- Final decoder stage concatenates with original grayscale input before final conv

This matches assignment requirement that final stage receives both deep features and original input information.

### 4.2 UNet training setup

- Same epoch count as previous CNN setting (default 25)
- Batch size: 100
- Same optimizer/loss framework as classification model

### 4.3 Why skip connections help (at least 2 reasons)

1. **Preserve fine details**: early-layer edge/texture information is directly reused in reconstruction stages.
2. **Improve gradient flow**: easier optimization of deeper networks due to shorter gradient paths.
3. **Better localization**: combines high-level semantic context with low-level spatial precision.

### 4.4 Performance discussion points

Executed UNet result (`epochs=25`):
- Final train loss: **1.2302**
- Final validation loss: **1.3032**
- Final validation accuracy: **50.4%**

Comparison to plain classification CNN:
- Validation loss improved from **1.5716 -> 1.3032**
- Validation accuracy improved from **41.7% -> 50.4%**
- UNet outputs are visually sharper at boundaries with better spatial consistency.

## 5) Visualizing Intermediate Activations

Use the pre-written visualization block in `colourization.ipynb`.

### 5.1 CNN activations

Generated activation files:
- `outputs/colourization_cnn/act0/conv1_out_0.png` to `conv5_out_0.png`

Observed pattern:
- Early layers emphasize local edges and simple intensity transitions.
- Intermediate layers show stronger response to horse/body regions.
- Deeper layers become more class/region selective with reduced fine-grained noise.

### 5.2 UNet activations

Generated activation files:
- `outputs/colourization_unet/act0/conv1_out_0.png` to `conv5_out_0.png`

Observed pattern:
- Skip-connected activations retain stronger boundary and texture signals in later stages.
- Decoder outputs preserve more local detail than the plain CNN.
- Final activations show a better balance of global semantic structure and local spatial precision.

## 6) Additional Theoretical Questions

### Q1) Pixel-level metrics vs human perception

Problem:
- Pixel losses (MSE, CE) optimize per-pixel correctness but do not reliably reflect perceptual realism.

Improvements:
1. Add **perceptual loss** (feature-space distances using pretrained CNNs, as in Johnson et al. 2016).
2. Include **human evaluation** or pairwise preference testing.
3. Use perceptual image metrics (e.g., SSIM/LPIPS-like ideas) in addition to pixel losses.
4. Evaluate color plausibility and consistency at object/region level, not only pixel-wise accuracy.

### Q2) Adapting 32x32-trained model to larger test images

Because these models are fully convolutional (no fully connected layers), they can be applied to larger images with care:

1. Feed larger grayscale images directly (if memory allows).
2. Ensure spatial dimensions are compatible with pooling/upsampling stages (multiples of downsampling factor), using padding/cropping if needed.
3. For very large inputs, run tiled/sliding-window inference and blend tile boundaries.
4. Optionally train/fine-tune with multi-scale data to improve robustness on larger resolutions.

## 7) References

1. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Johnson, J., Alahi, A., and Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution.

## 8) Reproducibility Checklist

- Regression notebook executed successfully (`colour_regression.executed.ipynb`).
- Classification + UNet notebook executed successfully (`colourization.executed.ipynb`).
- Generated outputs are available under:
  - `outputs/regression_cnn`
  - `outputs/colourization_cnn`
  - `outputs/colourization_unet`
- Activation visualizations generated for both CNN and UNet (`act0` folders).


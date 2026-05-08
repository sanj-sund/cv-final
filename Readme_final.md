# Deep Document Unwarping
ntroduction

The goal of this project was to design and train a deep learning pipeline capable of reconstructing a flattened and readable document image from a warped, folded, or crumpled page render. Unlike traditional document rectification problems that assume a flat planar surface, this project focuses on non-rigid geometric deformation. Curved pages, folds, wrinkles, and perspective distortions create spatial transformations that cannot be corrected using a simple homography or affine transformation. Instead, the system must learn a dense geometric mapping between the warped document image and the underlying flat document surface.

This project explores geometric deep learning techniques and differentiable image warping to address this problem. The implemented model predicts a dense UV coordinate field that describes how pixels from the warped document image should be mapped into a flattened coordinate space. The UV map is then used to reconstruct the rectified image through differentiable sampling.

The overall pipeline follows the structure:

Warped RGB Image → Encoder-Decoder Network → Predicted UV Coordinates → Differentiable Warping → Rectified Output

The system was implemented entirely in PyTorch using a pretrained convolutional backbone combined with a U-Net style decoder architecture. The final network predicts a two-channel UV deformation field that is later transformed into a sampling grid for differentiable image reconstruction using torch.nn.functional.grid_sample.

The project focuses primarily on geometry reconstruction rather than appearance enhancement or illumination correction. The objective is to learn the spatial deformation necessary to flatten the document while preserving document structure and readability.

Problem Formulation

Document rectification becomes significantly more difficult when the document surface is non-planar. In standard perspective correction problems, the transformation between the input image and the corrected output can often be approximated using a homography matrix. However, when a document contains folds, bends, or wrinkles, the mapping becomes spatially varying and nonlinear.

A crumpled document may contain:

Local folds
Curvature
Compression
Stretching
Perspective distortion
Severe geometric distortion

Because these deformations vary continuously across the image, the transformation must be modeled densely at the pixel level.

The implemented solution models document geometry through a UV coordinate representation. Each pixel in the warped image predicts a corresponding normalized coordinate in the flattened document space. This dense coordinate field represents the deformation required to reconstruct the original flat page.

The UV representation is advantageous because:

It provides direct geometric supervision
It separates geometry from appearance
It integrates naturally with differentiable warping
It avoids direct pixel generation

Instead of generating the rectified image directly, the model predicts geometric correspondence information. The final document reconstruction is then obtained through sampling operations.

Dataset

The dataset used for training consists of synthetically generated warped document renders. The dataset is organized into multiple folders containing:

RGB warped document renders
Ground truth flattened document textures
UV coordinate maps
Border masks

The RGB images represent crumpled or folded documents rendered under varying geometric conditions. The ground truth images represent the corresponding flattened document surfaces. The UV maps contain normalized coordinate information that defines the mapping between warped image pixels and flattened coordinates.

The UV coordinate maps are particularly important because they provide direct supervision for geometric learning. Each UV image contains two coordinate channels:

U coordinate
V coordinate

These coordinates define where each warped pixel belongs in the flattened document space.

The border masks identify foreground document regions and distinguish them from the background. The masks are used during training to ensure that the loss functions focus only on document pixels rather than background regions.

The dataset loader performs:

Image loading
UV loading
Mask loading
Resizing
Tensor conversion
Normalization

All images were resized to a lower training resolution to reduce memory usage and improve training efficiency.

Model Architecture

The implemented architecture follows an encoder-decoder design commonly used in dense prediction tasks such as segmentation, optical flow estimation, and depth prediction.

The network consists of:

Pretrained encoder
Multi-scale decoder
UV prediction head

The model predicts a dense two-channel UV coordinate map from the input RGB image.

Encoder

The encoder uses a pretrained ResNet34 backbone obtained through the timm library. Pretrained feature extractors were selected instead of training from scratch because pretrained models already contain strong low-level and mid-level visual representations learned from large-scale datasets such as ImageNet.

The encoder extracts hierarchical features at multiple spatial resolutions:

Low-level edge features
Mid-level texture features
High-level geometric features

Using a pretrained backbone improves:

Convergence speed
Stability
Feature quality
Generalization

The encoder outputs feature maps at multiple scales which are later used by the decoder through skip connections.

Decoder

The decoder reconstructs spatial detail while progressively increasing image resolution. A U-Net style decoder architecture was implemented because U-Net structures are highly effective for dense spatial prediction tasks.

The decoder uses:

Bilinear upsampling
Skip connections
Convolutional refinement blocks

At each stage:

Feature maps are upsampled
Concatenated with encoder skip features
Refined using convolutional blocks

The skip connections preserve fine spatial structure that may otherwise be lost during downsampling in the encoder.

The decoder progressively reconstructs:

Fold structure
Curvature patterns
Document boundaries
Geometric correspondence information

The final layer predicts a two-channel UV coordinate map using a convolution layer followed by sigmoid activation.

The sigmoid activation constrains predicted UV coordinates to the range [0,1].

UV Prediction

The predicted UV map represents normalized coordinates in flattened document space.

Each pixel predicts:

U coordinate
V coordinate

These coordinates describe where the corresponding warped pixel should map in the flattened image.

This approach treats document rectification as a geometric correspondence estimation problem rather than a direct image generation problem.

The UV representation is advantageous because:

It provides interpretable geometry
It separates deformation estimation from appearance reconstruction
It integrates naturally with differentiable warping

The UV predictions can also be visualized directly as color gradients, providing insight into the learned deformation field.

Differentiable Warping

After predicting the UV map, the system reconstructs the flattened document using differentiable inverse warping.

The UV coordinates are first converted from [0,1] range into normalized sampling coordinates in the range [-1,1], which is required by PyTorch’s grid_sample function.

The reconstruction pipeline follows:

Predict UV map
Convert UV map into sampling grid
Apply differentiable inverse warping using grid_sample

The differentiable warping operation allows gradients to propagate through the sampling process during training.

This enables end-to-end optimization of:

Geometry prediction
Reconstruction quality

The differentiable reconstruction stage is one of the central components of the pipeline because it directly connects geometric prediction with image reconstruction performance.

Loss Functions

The training objective combines multiple losses to encourage accurate geometric reconstruction and stable UV predictions.

The final loss function consists of:

UV reconstruction loss
Image reconstruction loss
Smoothness regularization
UV Reconstruction Loss

The primary supervision signal is a masked L1 loss between the predicted UV map and the ground truth UV coordinates.

This loss directly supervises geometric correspondence estimation.

The UV loss encourages:

Accurate spatial mapping
Correct deformation estimation
Stable geometric reconstruction

A masked loss was used to ensure that background pixels do not contribute significantly to optimization.

Image Reconstruction Loss

After differentiable warping, a masked L1 reconstruction loss is computed between:

Rectified image
Ground truth flattened image

This loss encourages the predicted UV coordinates to produce visually correct document reconstructions.

The reconstruction loss improves:

Document readability
Global structure
Alignment quality

Using reconstruction supervision alongside UV supervision improves stability and encourages better geometric consistency.

Smoothness Regularization

A smoothness loss penalizes abrupt local changes in neighboring UV coordinates.

This regularization encourages:

Continuous deformation fields
Reduced artifacts
Spatial coherence

Without smoothness constraints, predicted UV maps may contain noisy discontinuities that produce visual artifacts during reconstruction.

The smoothness loss was implemented using finite differences across horizontal and vertical coordinate directions.

Training Procedure

The model was trained using the Adam optimizer with a learning rate of:

1e-4

Training was performed for multiple epochs using mini-batch gradient descent.

The training loop performs:

RGB image loading
UV prediction
Differentiable reconstruction
Loss computation
Backpropagation
Parameter updates

The best-performing model checkpoint was saved automatically during training.

Training was performed on reduced-resolution images to improve training speed and reduce memory usage.

The implementation supports both:

GPU acceleration
CPU execution

GPU acceleration significantly improves training efficiency due to the computational cost of dense image reconstruction.

Inference Pipeline

After training, the inference pipeline automatically processes input images and saves:

Predicted UV visualizations
Rectified outputs
Comparison images

The comparison outputs display:

Original warped input
Reconstructed rectified output
Ground truth flattened image

These visualizations provide qualitative evaluation of geometric correction performance.

The inference script processes multiple images automatically without manual intervention.

Results

The trained model successfully learned coarse geometric document structure and produced partially flattened document reconstructions from heavily warped inputs.

The predicted UV fields captured:

Curvature
Fold direction
Spatial deformation
Global page geometry

The reconstructed outputs demonstrated:

Improved rectangular alignment
Increased readability
Reduced geometric distortion

Although some blur and local artifacts remained, the model consistently learned meaningful geometric correction behavior.

Loss values decreased steadily during training, indicating successful optimization and convergence.

The generated UV visualizations also demonstrated smooth and coherent coordinate predictions across document surfaces.

SSIM Metric

Structural Similarity Index Measure (SSIM) was used to evaluate reconstruction quality between:

Rectified output images
Ground truth flattened images

SSIM was selected because it evaluates structural similarity rather than raw pixel intensity differences. This makes it more appropriate for document reconstruction tasks where lighting conditions and local intensity variations may differ between images.

The final model achieved visually meaningful geometric correction while preserving major document structure and readability.

Limitations

Several limitations affected reconstruction quality.

First, training resolution was reduced due to computational and memory constraints. Higher-resolution training would likely improve fine text reconstruction and sharper geometric detail.

Second, the model used a convolutional encoder rather than a transformer-based backbone. Vision Transformers may capture larger-scale geometric relationships more effectively.

Third, only basic reconstruction losses were implemented. Additional losses such as:

SSIM loss
Perceptual loss
Edge consistency loss

could improve structural fidelity and visual sharpness.

Finally, severe folds and high-frequency wrinkles remain difficult to reconstruct accurately because extreme deformations may create ambiguous geometric correspondences.
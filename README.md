# Fast-Rcnn-and-SDD-
**Why Fast R-CNN?**

In R-CNN, object detection involves multiple steps that make it inefficient and slow. The key issues with R-CNN are:

**1. Multi-stage Training:**
   - Train a CNN on a large classification dataset.
   - Fine-tune the CNN with resized proposals on detection dataset classes (including the background class).
   - Train a binary classifier (SVM) for each class on the fully connected (FC) layer representation of proposals and train bounding box regressors.

**2. High Computational Cost:**
   - Feature extraction is performed separately for each object proposal in each image.
   - Features are written to disk before SVM and bounding box regression training, leading to high memory usage and slow processing.

**3. Slow Object Detection:**
   - Each image requires multiple forward passes through the CNN, making inference very slow.

To resolve these issues, Fast R-CNN was introduced.

**Fast R-CNN:**

Unlike R-CNN, where multiple forward passes are needed for each object proposal, Fast R-CNN optimizes this process:

**1. Single Forward Pass:**
   - Instead of processing each region proposal separately, the entire image is passed through a CNN only once to extract a feature map.

**2. Region of Interest (RoI) Pooling:**
   - The RoI Pooling layer maps the region proposals to the corresponding feature map locations.

**3. Feature Extraction & Prediction:**
   - Extracted region features are flattened and passed to fully connected (FC) layers for prediction.

**4. Classification & Bounding Box Regression:**
   - The model performs classification (e.g., object categories) and bounding box regression using the extracted features.

**5. Fixed-size Input for FC Layers:**
   - Since the RoI pooling layer ensures that all proposals are mapped to a fixed size, FC layers can efficiently process them without resizing each individually.


**Key Improvements Over R-CNN:**

1. **Single Forward Pass:** Feature extraction is performed once for the whole image rather than separately for each proposal.
2. **Efficient Training:** Eliminates the need for separate SVM training and bounding box regression as both are learned in a single network.
3. **Memory Efficient:** Avoids redundant storage of feature maps, reducing disk usage.
4. **Faster Inference:** Object detection is significantly faster compared to R-CNN.
**ARCHITECTURE OF FAST R-CNN**
![image](https://github.com/user-attachments/assets/9351d8ac-b71f-4f6d-9145-74987fdfcffb)






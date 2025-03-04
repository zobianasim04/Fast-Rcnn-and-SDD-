# **Fast R-CNN and SSD**

## **Why Fast R-CNN?**

In R-CNN, object detection involves multiple steps that make it inefficient and slow. The key issues with R-CNN are:

### **1. Multi-stage Training:**
- Train a CNN on a large classification dataset.
- Fine-tune the CNN with resized proposals on detection dataset classes (including the background class).
- Train a binary classifier (SVM) for each class on the fully connected (FC) layer representation of proposals and train bounding box regressors.

### **2. High Computational Cost:**
- Feature extraction is performed separately for each object proposal in each image.
- Features are written to disk before SVM and bounding box regression training, leading to high memory usage and slow processing.

### **3. Slow Object Detection:**
- Each image requires multiple forward passes through the CNN, making inference very slow.

### **Solution: Fast R-CNN**

Unlike R-CNN, where multiple forward passes are needed for each object proposal, Fast R-CNN optimizes this process:

1. **Single Forward Pass:** Instead of processing each region proposal separately, the entire image is passed through a CNN only once to extract a feature map.
2. **Region of Interest (RoI) Pooling:** The RoI Pooling layer maps the region proposals to the corresponding feature map locations.
3. **Feature Extraction & Prediction:** Extracted region features are flattened and passed to fully connected (FC) layers for prediction.
4. **Classification & Bounding Box Regression:** The model performs classification (e.g., object categories) and bounding box regression using the extracted features.
5. **Fixed-size Input for FC Layers:** Since the RoI pooling layer ensures that all proposals are mapped to a fixed size, FC layers can efficiently process them without resizing each individually.

### **Key Improvements Over R-CNN:**
✅ **Single Forward Pass:** Feature extraction is performed once for the whole image rather than separately for each proposal.
✅ **Efficient Training:** Eliminates the need for separate SVM training and bounding box regression as both are learned in a single network.
✅ **Memory Efficient:** Avoids redundant storage of feature maps, reducing disk usage.
✅ **Faster Inference:** Object detection is significantly faster compared to R-CNN.

---

## **Architecture of Fast R-CNN**

1. First, we generate the region proposal from a selective search algorithm. This selective search algorithm generates up to approximately 2000 region proposals.
2. These region proposals (RoI projections) combine with input images passed into a CNN network.
3. This CNN network generates the convolution feature map as output.
4. Then for each object proposal, a Region of Interest (RoI) pooling layer extracts the feature vector of fixed length for each feature map.
5. Every feature vector is then passed into twin layers of a softmax classifier and bounding box regression for classification and improving the bounding box position.
6. The model consists of a **single-stage** approach, unlike R-CNN's three-stage method.

![Fast R-CNN Architecture](https://github.com/user-attachments/assets/9351d8ac-b71f-4f6d-9145-74987fdfcffb)

---

## **CNN Network of Fast R-CNN**

Fast R-CNN uses pre-trained ImageNet networks (e.g., VGG-16), which contain:
- **5 Max-pooling layers**
- **5-13 Convolution layers**

### **Modifications to the Pre-trained Network:**
1. The network is modified to accept two inputs: the image and the list of region proposals.
2. The last pooling layer (e.g., 7×7×512) before fully connected layers is replaced by the **Region of Interest (RoI) pooling layer**.
3. The last fully connected and softmax layers are replaced by twin layers: **softmax classifier** and **K+1 category-specific bounding box regressor**.

![CNN Network](https://github.com/user-attachments/assets/7fe0ebd7-be6a-43e3-8f70-892ab90d8829)

**Output:**
- Image (size = 224×224×3 for VGG-16)
- Region proposals → Convolution feature map (size = 14×14×512 for VGG-16)

---

## **ROI Pooling Layer**

The feature map from the last convolutional layer is fed to an **ROI Pooling Layer** to extract a fixed-length feature vector from each region proposal.

✅ **How ROI Pooling Works?**
- Each region proposal is split into a **grid of cells**.
- The **max pooling** operation is applied to each cell to return a single value.
- All values from all cells represent the **feature vector**.
- If the grid size is **2×2**, the feature vector length is **4**.

---

## **Stages of Fast R-CNN**

### **1. Stage 1: Region Proposal Network (RPN)**

#### **1.1 Backbone Network**
1. The image passes through a convolutional network (like ResNet or VGG16).
2. Extracts important features from the image to create a feature map.

#### **1.2 Anchors**
1. Anchors are boxes of different sizes and shapes placed over points on the feature map.
2. Each anchor box represents a possible object location.
3. At every point on the feature map, anchor boxes are generated with different sizes and aspect ratios.

#### **1.3 Classification of Anchors**
1. The RPN predicts whether each anchor box is background (no object) or foreground (contains an object).
   - **Positive (foreground) anchors:** Boxes with high overlap with actual objects.
   - **Negative (background) anchors:** Boxes with little or no overlap with objects.

#### **1.4 Bounding Box Refinement**
1. The RPN refines the anchor boxes to better align them with the actual objects by predicting offsets (adjustments).

#### **1.5 Loss Functions**
1. **Classification Loss:** Helps the model decide if the anchor is background or foreground.
2. **Regression Loss:** Helps adjust the anchor boxes to fit the objects more precisely.

---

### **2. Stage 2: Object Classification and Box Refinement**

#### **2.1 Region Proposals**
1. After RPN, we get region proposals (refined boxes that likely contain objects).

#### **2.2 ROI Pooling**
1. The region proposals have different sizes, but the neural network needs fixed-size inputs.
2. ROI Pooling resizes all region proposals to a fixed size by dividing them into smaller regions and applying pooling.

#### **2.3 Object Classification**
1. Each region proposal is passed through a small network to predict the category (e.g., dog, car, etc.).
2. **Cross-entropy loss** is used to classify the objects into categories.

#### **2.4 Bounding Box Refinement (Again)**
1. The region proposals are refined again to better match the actual objects, using offsets.
2. This uses **regression loss** to adjust the proposals.

#### **2.5 Multi-task Learning**
1. The network in stage 2 learns both to predict object categories and refine bounding boxes at the same time.

---

### **3. Inference (Testing/Prediction Time)**

#### **3.1 Top Region Proposals**
1. During testing, the model generates a large number of region proposals, but only the top proposals (with the highest classification scores) are passed to the second stage.

#### **3.2 Final Predictions**
1. The second stage predicts the final categories and bounding boxes.

#### **3.3 Non-Max Suppression**
1. **Non-Max Suppression (NMS)** is applied to remove duplicate or overlapping boxes, keeping only the best ones.







# Fast-Rcnn-and-SDD-
##Why Fast R-CNN?##

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
First, we generate the region proposal from a selective search algorithm. This selective search algorithm generates up to approximately 2000 region proposals. These region proposals (RoI projections) combine with input images passed into a CNN network. This CNN network generates the convolution feature map as output. Then for each object proposal, a Region of Interest (RoI) pooling layer extracts the feature vector of fixed length for each feature map. Every feature vector is then passed into twin layers of softmax classifier and Bbox regression for classification of region proposal and improve the position of the bounding box of that object.The general architecture of Fast R-CNN is shown below. The model consists of a single-stage, compared to the 3 stages in R-CNN. It just accepts an image as an input and returns the class probabilities and bounding boxes of the detected objects.
![image](https://github.com/user-attachments/assets/9351d8ac-b71f-4f6d-9145-74987fdfcffb)
**CNN Network of Fast R-CNN**
Fast R-CNN is experimented with three pre-trained ImageNet networks each with 5 max-pooling layers and 5-13 convolution layers (such as VGG-16). There are some changes proposed in this pre-trained network, These changes are:
1.The network is modified in such a way that it two inputs the image and list of region proposals generated on that image.
2.Second, the last pooling layer (here (7*7*512)) before fully connected layers needs to be replaced by the region of interest (RoI) pooling layer.
3.Third, the last fully connected layer and softmax layer is replaced by twin layers of softmax classifier and K+1 category-specific bounding box regressor with a fully connected layer.
![image](https://github.com/user-attachments/assets/7fe0ebd7-be6a-43e3-8f70-892ab90d8829)
This CNN architecture takes the image (size = 224 x 224 x 3 for VGG-16) and its region proposal and outputs the convolution feature map (size = 14 x 14 x 512 for VGG-16).
**ROI POOLING LAYER**
The feature map from the last convolutional layer is fed to an ROI Pooling layer. The reason is to extract a fixed-length feature vector from each region proposal.
the ROI Pooling layer works by splitting each region proposal into a grid of cells. The max pooling operation is applied to each cell in the grid to return a single value. All values from all cells represent the feature vector. If the grid size is 2__×__2, then the feature vector length is 4.

## **Stages of Fast R-CNN**

### **1. Stage 1: Region Proposal Network (RPN)**

#### **1.1 Backbone Network**
1. The image passes through a convolutional network (like ResNet or VGG16).
2. This extracts important features from the image and creates a feature map.

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
2. ROI Pooling resizes all region proposals to a fixed size by dividing them into smaller regions and applying pooling, making them uniform.

#### **2.3 Object Classification**
1. Each region proposal is passed through a small network to predict the category (e.g., dog, car, etc.) of the object inside it.
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
1. A technique called **Non-Max Suppression (NMS)** is applied to remove duplicate or overlapping boxes, keeping only the best ones.






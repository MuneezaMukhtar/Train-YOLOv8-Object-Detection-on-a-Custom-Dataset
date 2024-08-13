# Train-YOLOv8-Object-Detection-on-a-Custom-Dataset


#### **Introduction**

YOLOv8 (You Only Look Once version 8) is the latest iteration of the YOLO series, known for its balance between speed and accuracy in object detection tasks. Training a YOLOv8 model on a custom dataset involves several stages: preparing the dataset, setting up the environment, configuring the model, and training it. 


#### **System Requirements**

Before starting, ensure your system meets the following requirements:

- **Operating System**: Linux, macOS, or Windows.
- **Python Version**: Python 3.8 or higher is required for compatibility with YOLOv8 and associated libraries.
- **Software Dependencies**:
  - **PyTorch**: The deep learning framework used by YOLOv8.
  - **OpenCV**: For image processing tasks.
  - **LabelImg**: A tool for labeling images.
  - **Other Python Libraries**: Includes numpy, matplotlib, and others depending on your specific needs.


#### **Installation**

 **Set up Python Environment**:
   
   - Install Python 3.8 or higher from [python.org](https://www.python.org/).
   - Create a virtual environment:
     ```bash
     python -m venv yolov8-env
     ```
   - Activate the virtual environment:
   - 
     - On Windows:
       ```bash
       yolov8-env\Scripts\activate
       ```

#### **Labeling Dataset with LabelImg**
Labeling your dataset is a critical step for training the YOLOv8 model. LabelImg is a popular tool for creating annotations.


#### **Using LabelImg to Label Your Dataset**

 **Install LabelImg**:
 
   - Install via pip:
     ```bash
     pip install labelImg
     ```
   - Alternatively, clone the repository and run it manually:
     ```bash
     git clone https://github.com/tzutalin/labelImg.git
     cd labelImg
     pip install -r requirements/requirements-linux-python3.txt
     python labelImg.py
     ```

 **Load Images**:
 
   - Open LabelImg and select the folder containing your images. You’ll see the images displayed one by one.

3. **Create Labels**:
   - Start drawing bounding boxes around the objects you want to detect. Assign a label (class) to each bounding box.

 **Save Annotations**:
 
   - Save the annotations in the YOLO format, which is required for training YOLOv8. Each image will have a corresponding `.txt` file containing the annotations.


**Label Images**

1. **Draw Bounding Boxes**:
 
   - Click and drag to draw a box around each object in the image. Ensure that the bounding box tightly surrounds the object.
   
2. **Assign Labels**:
 
   - After drawing a bounding box, type the label name (e.g., "dog", "car") in the text box provided. These labels should correspond to the classes you want your model to detect.

3. **Save Annotations**:
   
   - Save the annotations as `.txt` files in the YOLO format. Each line in the file should represent one bounding box in the format:
     ```
     <class_id> <x_center> <y_center> <width> <height>
     ```
     where all coordinates are normalized between 0 and 1 relative to the image dimensions.

#### **Annotation Formats**

YOLOv8 requires annotations in a specific format:
- **YOLO Format**: Each line in the annotation file represents a single bounding box and contains:
  - `class_id`: The numeric ID of the object class.
  - `x_center`, `y_center`: The normalized center coordinates of the bounding box.
  - `width`, `height`: The normalized width and height of the bounding box.

Example for an image with a single bounding box:
```
0 0.5 0.5 0.25 0.25
```


 **Training a Dataset with YOLOv8**

 **Prepare Your Dataset**

1. **Organize Files**:

   - Create a directory structure:
     ```
     dataset/
     ├── images/
     │   ├── train/
     │   ├── val/
     └── labels/
         ├── train/
         └── val/
     ```
   - Place your images in the `images/train` and `images/val` directories.
   - Place your annotation files in the `labels/train` and `labels/val` directories.

2. **Class Labels**:
 
   - Create a `data.yaml` file that defines your dataset paths and class names:
     ```yaml
     train: ../dataset/images/train
     val: ../dataset/images/val

     nc: 3  # number of classes
     names: ['class1', 'class2', 'class3']
     ```

#### **Install YOLOv8**

1. **Clone YOLOv8 Repository**:
   
   - Clone the YOLOv8 repository from GitHub:
     ```
     git clone https://github.com/ultralytics/yolov8.git
     cd yolov8
     ```

2. **Install YOLOv8**:
3. 
   - Install YOLOv8 and its dependencies:
     ```
     pip install -r requirements.txt
     pip install -e .
     ```

#### **Prepare Configuration File**

1. **Edit Configuration**:
   
   - Modify the `data.yaml` file to match your dataset paths, the number of classes (`nc`), and class names (`names`).
   - If necessary, create a custom model configuration file based on a predefined YOLOv8 model (e.g., yolov8s.yaml for YOLOv8-small).

2. **Hyperparameters**:
   
   - Adjust hyperparameters such as:
     - `batch_size`: Number of images processed at once.
     - `learning_rate`: Rate at which the model updates its weights.
     - `epochs`: Number of times the entire dataset is passed through the model during training.
   - Modify these in the model configuration file or pass them as arguments during training.

#### **Train the Model**

1. **Start Training**:
   
   Run the YOLOv8 training script with your custom configuration:
     
     ```
    yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
     ```
     
   The script will start training the model on your custom dataset.

2. **Monitor Training**:
   
   Training logs will display metrics like loss, accuracy, and learning rate. Monitor these metrics to ensure the model is learning correctly.

# F1 CURVE

![F1_curve](https://github.com/user-attachments/assets/76f0b508-8c15-4fef-ad20-1a6ca962c56b)


#### **Monitor Training**

1. **Loss and Accuracy Metrics**:
   
   - Keep an eye on the training and validation loss values. If the loss does not decrease or fluctuates wildly, consider adjusting hyperparameters.
   - Monitor accuracy-related metrics like mAP (mean Average Precision) to evaluate how well the model is detecting objects.

2. **Early Stopping**:
   
   - Implement early stopping if the model begins to overfit (i.e., performance on the validation set worsens while training performance improves). Early stopping will halt training when improvements are minimal.

# Confusion Matrix

![confusion_matrix](https://github.com/user-attachments/assets/b3955681-61e5-4fec-9faf-bdd50e27ebdc)

![P_curve](https://github.com/user-attachments/assets/8ff3fb8a-96ac-4270-9c81-fbef2bdf11ce)

![R_curve](https://github.com/user-attachments/assets/d298820b-85cd-4156-b255-252cc969530c)

![PR_curve](https://github.com/user-attachments/assets/0b3d2121-965c-48ae-8afa-8640741f7627)


#### **Evaluate and Test**

1. **Validation**:
   
   - After training, the model's performance on the validation set will be evaluated. This involves calculating metrics such as precision, recall, and mAP.
   - Check these metrics to see if the model meets your performance expectations.

2. **Testing**:
   
   - Once satisfied with validation results, evaluate the model on a separate test set. The test set should include data the model hasn't seen during training.
   - Use the following command:
     
     ```
     python val.py --weights yolov8_custom/best.pt --data data.yaml
     ```
   - Analyze the output metrics to understand the model's real-world performance.
     
  
   ![results](https://github.com/user-attachments/assets/6496fe74-3264-43e1-8a3e-3f5c8099a7e1)



#### **Inference**

1. **Run Inference**:
   
   To detect objects in new images, use the trained YOLOv8 model:

     ```
    yolo predict model=path/to/best_model.pt source=path/to/image_or_video
   
     ```
     
   The model will output images with bounding boxes drawn around detected objects.
   

   ![val_batch2_labels](https://github.com/user-attachments/assets/e9cb5a59-7bc1-4ac7-a511-6d93fc20638d)
   

3. **Visualize Results**:
   - Visualize or save the images with detections. YOLOv8 outputs the confidence score for each detected object, which can be displayed on the bounding boxes.
   

   ![val_batch2_pred](https://github.com/user-attachments/assets/000712cc-c7b7-4ff2-ae7e-1cc79ba7cbef)



#### **How It Works**

YOLOv8 operates by dividing an input image into a grid and assigning each grid cell the responsibility of predicting bounding boxes and class probabilities for objects. The model predicts these directly from the image in a single forward pass, making it highly efficient. YOLOv8 leverages convolutional neural networks (CNNs) to extract features from the image, which are then processed to predict the location

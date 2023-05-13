#      Yolo-Papaya: A Papaya Fruit Disease Detector and Classifier Using CNNs and Convolutional Block Attention Modules

Early detection of diseases in fruits is essential to mitigate production losses and ensure the quality and health of the fruits offered to end consumers. Due to the peculiarities of this domain, such as the high intra-class variability and the lack of public datasets with the required number of examples for use in state-of-the-art neural networks, disease detection in fruits is still a challenging problem for computer vision. In this work, we propose Yolo-Papaya, a robust disease detector based on YoloV7, which uses a Convolutional Block Attention Module (CBAM) mechanism to reduce information redundancy between channels and focus on the most relevant regions of the feature map. We also present a dataset of papaya fruit images with over 23,000 examples divided into nine disease classes with multi-class and multi-instance annotations. The proposed detector achieved a robust improvement compared to other tested neural networks, with an average mAP score of 86.4%, without significant increases in inference time and neural network memory consumption.

##
Published work:
https://www.mdpi.com/2079-9292/12/10/2202/htm

### Requirements
- Ubuntu 20.04
- Cuda 11.7
- Cudnn 8.0
- OpenCV 4.5

### Requirements Extras
```
#### Base ----------------------------------------
- matplotlib>=3.2.2
- numpy>=1.18.5,<1.24.0
- opencv-python>=4.1.1
- Pillow>=7.1.2
- PyYAML>=5.3.1
- requests>=2.23.0
- scipy>=1.4.1
- torch>=1.7.0,!=1.12.0
- torchvision>=0.8.1,!=0.13.0
- tqdm>=4.41.0
- protobuf<4.21.3

#### Logging -------------------------------------
 - tensorboard>=2.4.1

#### Plotting ------------------------------------
 - pandas>=1.1.4
 - seaborn>=0.11.0

#### Extras --------------------------------------
 - psutil  # system utilization
 - thop  # FLOPs computation
 
 ### Install
     apt update
     apt install -y zip htop screen libgl1-mesa-glx
     pip install seaborn thop
     pip install -r requirements.txt
```
     
 	- Our template is based on the original Yolov7 framework available at https://github.com/WongKinYiu/yolov7. 


## Results

- Expected result for the test set


<img src=https://github.com/jhony2507/Sisfrutos-Papaya/blob/main/img/ResultPerClass.png height=671 e width=891>



<img src=https://github.com/jhony2507/Sisfrutos-Papaya/blob/main/img/Results.png height=229 e width=567>


## How to use

### Inferences
- To test an image: 
      
      python3 detect.py  --source samples/TT000001-6.jpg --weights weights/w-yolo-papaya.pt --conf_thres=0.15 --img 448
- To test all images in a folder:
      
      python3 detect.py  --source samples/TT000001-6.jpg --weights weights/w-yolo-papaya.pt --conf_thres=0.15 --img 448
     
- Optional parameters:

   -project [PROJECT] : Defines the path of the main folder where the results will be saved (default = runs/detect)

   -name [name] : Defines the name of the folder (below PROJECT) where the experiments will be saved (default = exp)
   
   -conf_thres [value] : Confidence threshold


- To calculate the mAP of all classes in the test or validation set:

      python3 test.py --task=test --data data/papaya-data.yaml --img 448 --device 0 --weights weights/w-yolo-papaya.pt 

 where,   
```
    -task    : defines the set to be used in the experiment (test or val)
    -data    : file with the classes and the path location of the test or validation file
    -weights : weights file
```
## Training

- Using pre-trained weights: 

      python3 train.py  --cfg cfg/CBAM.yaml --data data/papaya-data.yaml --hyp cfg/hyperparamet.papaya.yaml --img 448  --name xxx  --weights weights/w-yolo-papaya.pt
     
 where,   
 
```
  --cfg : Indicates the network configuration file
  --data : Indicates the file with the classes and location of the training/validation and test files
  --hyp  : file with hyperparameters
  --img  : image size
  --name : Name of the new network
  --weights : used weights file
```
    
- For training from scratch: 

      python3 train.py  --cfg cfg/CBAM.yaml --data data/papaya-data.yaml --hyp cfg/hyperparamet.papaya.yaml --img 448  --name xxx  --weights ``
     
## Datasets

Our dataset consists of 23,158 examples spread over 17,949 images. File naming is as follows:

XXNNNNNN-C-C....jpg

where:
- XX     : Indicates whether the image belongs to the training (TR), test (TT) or validation (VA) set;
- NNNNNN : Sequential number
- -C-C   : Code of the class(es) that appear in the image.

Example:

- TT000009-8.jpg     : Image of the TEST set with an instance of the "Scar" class (8)
- TT000582-8-7-7.jpg : Image of the TEST set with one occurrence of the "Scar" class (8) and two occurrences of the "Black Spot " class (7)

The dataset is unbalanced, and follows the following distribution across classes:


<img src=https://github.com/jhony2507/Sisfrutos-Papaya/blob/main/img/Classes.png height=400 e width=800>


### Annotations

In an effort to broaden the accessibility of the new dataset to a wider range of re-searchers in the field, we have included annotations in two widely employed standards for state-of-the-art (SOTA) classifiers in the object detection task.

#### Txt Format:
 - Each image (.jpg) has its respective .txt file. For example, image TR00001-4.jpg is related to file TR00001-4.txt; 
 - Each line of the .txt file describes an object that appears in the respective image; 
 - The content of each line contains the following data:
   - class x-center y-center width height
 
 where,   
 
 - class   : Id with the object’s class
 - x-center: coordenada do eixo x do ponto central do object (in relation to image size)
 - y-center: coordenada do eixo y do ponto central do object (in relation to image size)
 - width   : width od object
 - height  : height od object
 
  The figure below shows an example. (a)-Image TR00001-4.jpg, (b)-Contents of the TR00001-4.txt file. The first line indicates the location and size of a 'Papaya' object (Id: 0), the second line indicates the coordinates and size of a 'chocolate spot' object (id:4). 

<img src=https://github.com/jhony2507/Sisfrutos-Papaya/blob/main/img/Exemplo_anotacao_TXT.png height=480 e width=400>
      
#### COCO Format      
The instances annotations of each object is composed of a basic data structure that contains a series of fields with information about the image, annotations and classes (categories). The storage in the form of structured records allows using a single json file to store the annotations of an entire data set.

  The figure below show partial annotations in Coco format for image TR00001-4.jpg 
  
<img src=https://github.com/jhony2507/Sisfrutos-Papaya/blob/main/img/Exemplo_anotacao_coco.png height=780 e width=550>

 
### License to use and download
The Sisfrutos dataset is made available free of charge to academic and non-academic entities, such as research, teaching, scientific publications or personal experimentation, on a non-commercial basis. The use of this dataset, in whole or in part, is expressly prohibited for commercial purposes.
Permission iLicense to use and downloads granted to use the data as long as you agree to our license terms:

That the dataset is made available "AS IS" without express or implied warranty. In our work, every effort has been made to ensure maximum accuracy, however, we do not accept any responsibility for errors or omissions.

That you include a reference to the Papaya Dataset in any work that makes use of the dataset, in whole or in part.

That you do not distribute this dataset or modified versions. It is permitted to distribute derivative works as long as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).

That you may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, creating commercial applications with trained weights with the model, or otherwise using the data for the purpose of obtaining a commercial gain.

Send an email to artsoft.lucas@terra.com.br informing:
- Name:
- Country: 
- Office:
- Linked institution:
- Briefly describe how the dataset will be used:

#### Download

-  Dataset Train (~1 Gb)        : https://drive.google.com/drive/folders/1JMY7FYfRXf--5DqjR0gRFW1dbNvbdB82?usp=share_link
-  Dataset Test (~110 Mb)       : https://drive.google.com/drive/folders/170H-cycnQK5wmE9jAgWAfqOH_W3RT_OA?usp=share_link
-  Dataset Validation (~110 Mb) : https://drive.google.com/drive/folders/1SQZZYUf_xuz8Mw1gew3MqxAFqrJ5g_Ht?usp=share_link



#      Yolo-Papaya: A Papaya Fruit Disease Detector and Classi-fier Using CNNs and Convolutional Block Attention Modules

"A detecção precoce de doenças em frutas é essencial para mitigar perdas na produção e garantir a qualidade e a saúde dos frutos oferecidos aos consumidores finais. Em função das peculiaridades desse domínio, como, por exemplo, a grande variabilidade intraclasse e a falta de conjuntos de dados públicos com a quantidade de exemplos necessários para a utilização em redes neurais de ponta, a detecção de doenças em frutas ainda é um problema desafiador para a visão computacional. Neste trabalho, propomos o Yolo-Papaya, um detector de doenças robusto baseado na YoloV7, que utiliza um mecanismo de atenção Convolutional Block Attention Module (CBAM) para reduzir a redundância de informações entre canais e focar nas regiões mais relevantes do mapa de recursos. Também apresentamos um conjunto de dados de imagens de frutas de mamão com mais de 23.000 exemplos divididos em nove classes de doenças com anotações multi-classes e multi-instâncias. O detector proposto obteve uma melhoria robusta em relação a outras redes neurais testadas, com um escore mAP médio de 86,4%, sem aumentos significativos no tempo de inferência e no consumo de memória da rede neural."


### Requirements
- Nossa implementaçÃo usa como base a estrutura original da Yolov7 disponibilizado em https://github.com/WongKinYiu/yolov7. As principais alterações estão centralizadas nos seguintes arquivos:
  - yolo.py : Chamadas para o modolo de atenção CBAM
  - models/common.py : Implementação do modulo CBAM
  - papaya-cbam, papaya-data, papaya-hyp : Estrutura rede criada, dados, e parametros.
 
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
## Weights and results of the tested models


<img src=https://github.com/jhony2507/Base_doencas_mamao/blob/main/classes.png height=300 e width=450>



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
     
     
-  Description: Complete base composed of 23.153 images with the following class distributions:
<img src=https://github.com/jhony2507/Base_doencas_mamao/blob/main/classes.png height=300 e width=450>


-  Size file   : ~1gb
-  
### Annotations

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

<img src=https://github.com/jhony2507/Base_doencas_mamao/blob/main/Exemplo_anotacao_TXT.png height=380 e width=300>
      
#### COCO Format      
The instances annotations of each object is composed of a basic data structure that contains a series of fields with information about the image, annotations and classes (categories). The storage in the form of structured records allows using a single json file to store the annotations of an entire data set.

  The figure below show partial annotations in Coco format for image TR00001-4.jpg 
  
<img src=https://github.com/jhony2507/Base_doencas_mamao/blob/main/Exemplo_anotacao_coco.png height=780 e width=550>

 
### License to use and download
The Sisfrutos dataset is made available free of charge to academic and non-academic entities, such as research, teaching, scientific publications or personal experimentation, on a non-commercial basis. The use of this dataset, in whole or in part, is expressly prohibited for commercial purposes.
Permission iLicense to use and downloads granted to use the data as long as you agree to our license terms:

That the dataset is made available "AS IS" without express or implied warranty. In our work, every effort has been made to ensure maximum accuracy, however, we do not accept any responsibility for errors or omissions.

That you include a reference to the Papaya Dataset in any work that makes use of the dataset, in whole or in part.

That you do not distribute this dataset or modified versions. It is permitted to distribute derivative works as long as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data).

That you may not use the dataset or any derivative work for commercial purposes such as, for example, licensing or selling the data, creating commercial applications with trained weights with the model, or otherwise using the data for the purpose of obtaining a commercial gain.

Send an email to artsoft.lucas@terra.com.br informing:
- Name:
- Office:
- Linked institution:
- Search where the dataset will be used.

[Sisfrutos Dataset](https://drive.google.com/drive/folders/1GhCxUPzlfXBJRIXsuwiDwkZY8aNduu1_)

[Samples for inference](https://drive.google.com/drive/folders/151dIcYDCCE-e-TZC2sTSLh2G0LEDvoe9)



# Comments

The poor performance in detecting the Black Spot class surprised us, as it has very striking visual characteristics, which should favor the neural network. Upon further investigation, we detected that the poor performance is related to how, in some cases, the disease was noted in the ground truth. Black spot can occur in several disjoint regions of the fruit (several instances of the “Black Spot” class), but often the evaluator marks a large region of the fruit as being a single instance of that class. This makes the network detect multiple instances of this class outside the ground truth bound box. The images below exemplify this situation.

We are reviewing the notes of this class, and this review, as well as the results will be available on the project page. 

<img src=https://github.com/jhony2507/Base_doencas_mamao/blob/main/wrong_notes.png height=400 e width=650>

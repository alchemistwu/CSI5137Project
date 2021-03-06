# CSI5137Project
## Dependency:
* `tensorflow-gpu==2.3`
* `numpy==1.18.5`
* `matplotlib==3.3.2`

## To rapidly replicate the work:
### Download trained weights:
* [Download the trained model](https://drive.google.com/file/d/1F0-pGaj47s0vCFJMCIvW-A0J4sSsR2Bh/view?usp=sharing)
* Unzip it and replace the original `model` folder if there is one.
### Download the processed data
* [Download the processed data](https://drive.google.com/file/d/1UUHAg1yqaca2X4MruYS2aIBUjyl9Ooln/view?usp=sharing)
* Unzip it and replace the original `data` folder if ther is one.
## To run the experiment from scratch:
### Preparing the data:
* Download and extract dataset first: `https://www.kaggle.com/c/malware-classification/data`
* Convert binary files to jpg images:
* Use outstanding row based texture partitioning method: `python data_utils.py -ori_dir [directory to training folder in downloaded dataset] 
-ori_csv [directory to training csv in downloaded dataset] 
-train_dir [for saving training data] -test_dir [for saving test data]`
  
* Use continous sliding window method:`python data_utils.py -ori_dir [directory to training folder in downloaded dataset] 
-ori_csv [directory to training csv in downloaded dataset] 
-train_dir [for saving training data] -test_dir [for saving test data] -window`
  
* Note that: The outstanding row based method and the continous sliding window method should share same directories. 
  Since a postfix "row" will be automatically added to the directory you put if the outstanding row based method is selected.
  
### Training
* To train a DenseNet model with outstanding row(similarly you do not need to change the directories for training
  and testing, if "-row" is specified, a postfix "row" will be automatically added to the directories):
* `python train.py -train_dir [training dir] -test_dir [test dir] -row -model dense`
* To train a DenseNet model with continous sliding window:
* `python train.py -train_dir [training dir] -test_dir [test dir] -model dense`  
* You may also specify which model to train by `-model`:
* Options are: 
  * `res`: resnet-50
  * `vgg`： vgg16
  * `googLeNet`: Inception-v3
  * `dense`: DenseNet121
  * `mobile`: MobileNet V2
* Using multiple GPU by default, to use single gpu by specifying `-single_gpu`
* To initialize the model by ImageNet pretrained weights, specify `-pretrain`
* For more options:
`python train.py -h`

### Testing
* `-pretrain`, `-model`, `-row` are also available in testing setups.
* A quick example: `python predict.py -train_dir [training dir] -test_dir [test dir] -row -model dense`

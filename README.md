# CSI5137Project
## Dependency:
* `tensorflow-gpu==2.3`
* `numpy==1.18.5`
* `matplotlib==3.3.2`
## To run the experiment:
* Download and extact dataset first: `https://www.kaggle.com/c/malware-classification/data`
* Convert binary files to jpg images:
* `python data_utils.py -ori_dir [directory to training folder in downloaded dataset] 
-ori_csv [directory to training csv in downloaded dataset] 
-train_dir [for saving training data] -test_dir [for saving test data]`
* To train a model:
* `python train.py -train_dir [training dir] -test_dir [test dir]`
* You may also specify which model to train:
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

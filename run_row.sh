conda activate tfnight
python train.py -model res -row
python train.py -model res -pretrain -row
python train.py -model vgg -row
python train.py -model vgg -pretrain -row
python train.py -model googLeNet -pretrain -row
python train.py -model googLeNet -row
python train.py -model mobile -row
python train.py -model mobile -pretrain -row
python train.py -model dense -row
python train.py -model dense -pretrain -row


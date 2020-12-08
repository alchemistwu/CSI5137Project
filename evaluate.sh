source ~/anaconda3/etc/profile.d/conda.sh
conda activate tfnight
python predict.py -model googLeNet -pretrain -row
python predict.py -model googLeNet -row
python predict.py -model mobile -row
python predict.py -model mobile -pretrain -row
python predict.py -model dense -row
python predict.py -model dense -pretrain -row
python predict.py -model res -row
python predict.py -model res -pretrain -row
python predict.py -model vgg -row
python predict.py -model vgg -pretrain -row

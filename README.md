# ECG Analysis

Download & Unzip the processed data: `https://drive.google.com/file/d/1Vn0i9eHE0tAGllFeW45jsidczTn7HJ1z/view?usp=sharing`

Install the requirements: 

`pip install -r requirements.txt`

Train: 

`python train.py --data_path dataset/cinc2017 --batch_size 32 --max_epochs 100 --model_barebone resnet50`

Run the tensorboard:

`tensorboard --bind_all --logdir logs`

Generate the explain: 

`python explain.py --data_path dataset/cinc2017 --ckpt_path logs/ckpt/epoch=19-val_loss=0.41-val_f1=0.79.ckpt`
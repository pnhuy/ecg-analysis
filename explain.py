import argparse
import json
import os
import torch
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from data.images.images import PtbXlDataModule
from models.images.images import ImageClassifier
from train_poincare import set_seed

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def main(args):
    set_seed(seed=args.seed)

    train_dir = os.path.join(args.data_path, 'processed/train')
    train_label = os.path.join(args.data_path, 'processed/y_train.csv')
    val_dir = os.path.join(args.data_path, 'processed/val')
    val_label = os.path.join(args.data_path, 'processed/y_val.csv')

    datamodule = PtbXlDataModule(
        train_dir=train_dir,
        train_label=train_label,
        val_dir=val_dir,
        val_label=val_label,
        # test_dir=test_dir,
        # test_label=test_label,
        batch_size=args.batch_size
    )

    val_data = datamodule.val_dataloader()
    val_len = len(val_data)

    classes = datamodule.train_dataset.labels.columns
    saved_model = ImageClassifier.load_from_checkpoint(
        args.ckpt_path,
        classes=classes,
        barebone='resnet50', # 'vit_b_16'
        learning_rate=5e-4,
        loss_type='bce')

    model = saved_model.network.to(DEVICE)
    target_layers = [model.layer4[-1]]
    

    # Get Prediction
    predictions = dict()
    for idx, data in tqdm(enumerate(val_data), total=val_len, desc='Get Prediction'):
        input_tensor = data['image'].to(DEVICE)
        label = data['label'].cpu().numpy()
        label_id = label.argmax()

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)

        for b in range(probs.shape[0]):
            filename = f"batch-{idx:03d}-{b:03d}.png"
            prob = probs[b]
            pred_id = prob.argmax().item()
            prob = prob.max().item()

            predictions[filename] = dict(
                label=classes[label_id],
                pred=classes[pred_id],
                prob=prob
            )

    os.makedirs(os.path.join(args.log_dir, 'cam_output'), exist_ok=True)
    with open(os.path.join(args.log_dir, 'cam_output', 'predictions.json'), 'w') as f:
        json.dump(predictions, f)

    for CAM in (GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad):
        output_dir = os.path.join(args.log_dir, 'cam_output', CAM.__name__)
        os.makedirs(output_dir, exist_ok=True)
        cam = CAM(model=model, target_layers=target_layers, use_cuda=True)

        for idx, data in tqdm(enumerate(val_data), total=val_len, desc=CAM.__name__):
            input_tensor = data['image']
            batch_grayscale_cam = cam(input_tensor=input_tensor, targets=None)

            for b in range(probs.shape[0]):
                filename = f"batch-{idx:03d}-{b:03d}.png"
                fp = os.path.join(output_dir, filename)
                if os.path.isfile(fp):
                    continue
                preds = predictions[filename]

                grayscale_cam = batch_grayscale_cam[b, :]
                rgb_img = input_tensor[b].permute(1, 2, 0).detach().cpu().numpy()
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                plt.imshow(visualization)
                plt.axis('off')
                plt.title(f"Label: {preds['label']} - Pred: {preds['pred']} - Prob = {preds['prob']:.2f}")
                plt.savefig(fp)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    

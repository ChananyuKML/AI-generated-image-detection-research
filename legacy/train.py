import torch.optim as optim
import torch
import torch.nn as nn
import argparse
from utils import train_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.vit import *
from networks.resnet import *
from networks.hog_cnn import *
from networks.learned_filters import LearnableFilterCNN128
import os
import matplotlib.pyplot as plt
import pathlib


def train_model(model,model_name, transform, dataroot, dataset, train_loader, valid_loader, num_epochs=30, lr=1e-4, device="cuda"):
	for split_name, loader in (("training", train_loader), ("validation", valid_loader)):
		if len(loader.dataset) == 0:
			raise ValueError(f"The {split_name} dataset is empty under '{dataroot}'. Check the image folders and selected loader.")
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		model = nn.DataParallel(model)
	model.to(device)
	print("torch version : ", torch.__version__)
	print("cuda available? : ", torch.cuda.is_available())
	print("cuda version: ",torch.version.cuda)
	print("cuda device count: ", torch.cuda.device_count())
	device_ids=list(range(torch.cuda.device_count()))
	print("cuda device id: ", *device_ids, sep=", ")
	out_path = f'checkpoints/{model_name}_{transform}_{dataset}'
	print(lr)
	print(dataroot)
	path = pathlib.Path(out_path)
	train_loss_hist = []
	val_loss_hist = []
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	best_val_loss = float('inf')
	if path.exists() and path.is_dir():
		print("Directory exists:", path)
	else:
		path.mkdir(parents=True, exist_ok=True)
	for epoch in range(num_epochs):
		model.train()
		train_loss = 0
		correct = 0
		iter = 1

		for data, labels in train_loader:
			data, labels = data.to(device), labels.to(device).float().unsqueeze(1)
			optimizer.zero_grad()
			outputs = model(data)
			loss = criterion(outputs, labels)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			pred = (torch.sigmoid(outputs) > 0.5).float()
			train_loss += loss.item() * data.size(0)
			correct += (pred  == labels).sum().item()
			if iter%10==1:
				print(f"currently training at iteration : {iter} / {len(train_loader)}", end="\r", flush=True)
			iter += 1

		acc = correct / len(train_loader.dataset)
		train_loss /= len(train_loader.dataset)
		print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {acc:.4f}")

		# Validation
		model.eval()
		val_loss = 0.0
		val_correct = 0
		with torch.no_grad():
			for inputs, labels in valid_loader:
				inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
				outputs = model(inputs)
				loss = criterion(outputs, labels)        
				val_loss += loss.item() * inputs.size(0)
				pred = (torch.sigmoid(outputs) > 0.5).float()
				val_correct += (pred == labels).sum().item()
			val_acc = val_correct / len(valid_loader.dataset)
			val_loss /= len(valid_loader.dataset)
			print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				torch.save(model.state_dict(), f'{out_path}/ckpt_{epoch+1}.pth')
				print(f"Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")
			train_loss_hist.append(train_loss)
			val_loss_hist.append(val_loss)
	return train_loss_hist, val_loss_hist

if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    opt = train_opt.TrainOptions().parse()
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        print(f"Random seed: {opt.seed}")
    if opt.network in ("learned_filter_cnn", "fixed_filter_cnn") and opt.transform != "gray_center_224":
        raise ValueError(f"--network {opt.network} requires --transform gray_center_224 (raw grayscale input); gradient transforms already apply the filters.")
    dataroot = f"{opt.dataroot}/{opt.dataset}"
    print(f"Training using dataset {dataroot}")
    match opt.network:
        case "res18_rgb":
            model = Resnet18()
        case "res18_rgb_unlocked":
            model = Resnet18_unlocked()
        case "dual_vit":
            model = DualViTv2()
        case "pret_vit":
            model = ViTRGB()
        case "pret_vit_2ch":
            model = ViT2ch()
        case "dual_cnn":
            model = DualCNN()
        case "dual_cnn_pool":
            model = DualCNN_pooling()
        case "dual_cnn_128":
            model = DualCNN_128()
        case "learned_filter_cnn":
            model = LearnableFilterCNN128(trainable=True)
        case "fixed_filter_cnn":
            model = LearnableFilterCNN128(trainable=False)
        case "dual_cnn_128_6ch":
            model = DualCNN_128(in_channels=6)
        case "dual_cnn_128_5ch":
            model = DualCNN_128_5ch()
        case "hog_cnn":
            model = HogCNN()
        case "res18_2ch":
            model = Resnet18_2CH()
        case "res18_2ch_unlocked":
            model = Resnet18_2CH_unlocked()
        case "2ch_vit":
            model = ViT2CH()
        case "pretrain_vit":
            model = PretrainViT()
        case "train_fil":
            model = DualCNN_128_trainFilter()
        case "HogHist8D":
            model = HogHist8D()
        case "_":
            print("Unknown Loader")

    match opt.transform:
        case "hog_224":
            transform = custom_transforms.hog_224
        case "hog_224_vit":
            transform = custom_transforms.hog_224_vit
        case "hogfft_224":
            transform = custom_transforms.hogfft_224
        case "hogfft_256":
            transform = custom_transforms.hogfft_256
        case "hog_256":
            transform = custom_transforms.hog_256
        case "hog_hist":
            transform = custom_transforms.hog_hist
        case "hog_hist_8D":
            transform = custom_transforms.hog_hist_8D
        case "HoG_224":
            transform = custom_transforms.HoG_224
        case "rgb_224":
            transform = custom_transforms.rgb_224
        case "gray_224":
            transform = custom_transforms.gray_224
        case "gray_center_224":
            transform = custom_transforms.gray_center_224
        case "grad_224":
            transform = custom_transforms.grad_224
        case "grad_224_crop":
            transform = custom_transforms.grad_224_crop
        case "raw_grad_224":
            transform = custom_transforms.raw_grad_224
        case "rgb_grad_224":
            transform = custom_transforms.rgb_grad_224
        case "prewitt_grad_224":
            transform = custom_transforms.prewitt_grad_224
        case "scharr_grad_224":
            transform = custom_transforms.scharr_grad_224
        case "grad_224_5ch":
            transform = custom_transforms.grad_224_5ch
        case "dxdy_224":
            transform = custom_transforms.dxdy_224
        case "_":
            print("Unknown Loader")

    match opt.loader:
        case "createByDir":
            train_loader, val_loader, _ = create_dataloader.createByDir(dataroot, opt.batch_size, transform=transform, num_workers=opt.num_workers)
        case "createByDir2":
            train_loader, val_loader = create_dataloader.createByDir2(dataroot, opt.batch_size, transform=transform, num_workers=opt.num_workers)
        case "gen_image":
            try:
                train_loader, val_loader = create_dataloader.gen_image(dataroot, opt.batch_size, transform=transform, num_workers=opt.num_workers)
            except ValueError as error:
                raise ValueError(f"Cannot create gen_image loaders under '{dataroot}': {error}. Expected train/ai, train/nature, val/ai, and val/nature image folders.") from error
        case "_":
            print("Unknown Loader")
    print(f"dataset created\ndevice={device}")
    train_hist, val_hist = train_model(model=model, model_name=opt.network, dataset=opt.dataset, dataroot=dataroot, transform=opt.transform, train_loader=train_loader, valid_loader=val_loader, num_epochs=opt.epochs, lr=opt.lr, device=device)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt.network}_{opt.transform} Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pathlib.Path('output/graph').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'output/graph/{opt.network}_{opt.transform}_{opt.dataset}_{opt.epochs}.png')
    if plt.get_backend().lower() == 'agg':
        plt.close()
    else:
        plt.show()

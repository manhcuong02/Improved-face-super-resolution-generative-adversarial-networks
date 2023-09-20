from dataset import TrainDataset
from train import train
import yaml
from torchvision import transforms as T
from torch.utils.data import DataLoader

with open('config.yaml', "r") as f:
        config = yaml.full_load(f) 

transform = T.Compose(
    [
        T.ToTensor()
    ]
)


train_dataset = TrainDataset(gt_images_dir = config['data_root_dir'], transform = transform)
train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers'])

gen_model, train_model = train(train_loader, device='cuda')
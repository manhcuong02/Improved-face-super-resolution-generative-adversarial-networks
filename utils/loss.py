import torch
import torch.nn as nn

from torchvision.models import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms as T

class PerceptualLoss(nn.Module):
    def __init__(self,
        features_nodes = ["features.24"],
        vgg_pretrained =  True,
        device = 'cpu'
    ):
        super(PerceptualLoss, self).__init__()
        
        if isinstance(device, str):
            if (device == 'cuda' or device == 'gpu') and torch.cuda.is_available():
                device = torch.device(device)
            else:
                device = torch.device('cpu')

        
        if vgg_pretrained:
            self.model = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).to(device)
        else:
            self.model = vgg19().to(device)
            
        self.features_nodes = features_nodes
        
        self.feature_extractor = create_feature_extractor(self.model, return_nodes = features_nodes)
        
        self.feature_extractor.eval()
        
        self.euclide_loss = nn.MSELoss()
        
    def transform(self, x: torch.Tensor):
        '''x: batch, channels, height, width'''
        return T.Compose(
            [
                T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]
        )(x)
        
    def forward(self, sr_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
        assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
        
        device = sr_tensor.device
        
        sr_tensor = self.transform(sr_tensor)
        gt_tensor = self.transform(gt_tensor)
        
        sr_features = self.feature_extractor(sr_tensor)
        gt_features = self.feature_extractor(gt_tensor)
        
        losses = []
        
        for layer in self.features_nodes:
            losses.append(
                self.euclide_loss(sr_features[layer], gt_features[layer])
            )
        
        if len(losses) == 1:
            return losses[0].to(device)
        else:
            return torch.Tensor(losses).to(device)
        
class PixelWiseLoss(nn.Module):
    def __init__(self):
        super(PixelWiseLoss, self).__init__()
        
        self.loss_func = nn.MSELoss()
        
    def forward(self, in_tensor, target_tensor) -> torch.Tensor:
        
        loss = self.loss_func(in_tensor, target_tensor)
        
        return loss
    
class AdversarialLoss(nn.Module) :
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        
        self.loss_func = nn.BCELoss()
        
    def forward(self, in_tensor, target_tensor) -> torch.Tensor:
        loss = self.loss_func(in_tensor, target_tensor)
        
        return loss
    
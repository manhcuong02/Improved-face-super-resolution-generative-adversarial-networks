import torch
import yaml
from torch import nn
from torch.distributions import Uniform
from torch.optim import Adam
from tqdm import tqdm

from models import Discriminator, Generator
from utils.loss import AdversarialLoss, PerceptualLoss, PixelWiseLoss


def train(train_loader, device = 'cpu'):
    
    if isinstance(device, str):
        if (device == 'gpu' or device == 'cuda') and torch.cuda.is_available():
            device = torch.device('cuda')
        else: 
            device = torch.device('cpu')
    
    with open('config.yaml', "r") as f:
        config = yaml.full_load(f) 
    
    #model 
    gen_model = Generator().to(device)
    dis_model = Discriminator().to(device)
    
    # criterion 
    pixel_criterion = PixelWiseLoss()
    content_criterion = PerceptualLoss(device = device)
    adversarial_criterion = AdversarialLoss()
    
    # optimizer 
    lr = config['lr']
    betas = config['betas']
    dis_optim = Adam(dis_model.parameters(), lr = lr, betas = betas, eps = 0.0001)
    gen_optim = Adam(gen_model.parameters(), lr = lr, betas = betas, eps = 0.0001)
    
    num_steps = len(train_loader)
    iterator = iter(train_loader)
    count_steps = 1  
    
    adversarial_weight = config['adversarial_weight']
    epochs = config['epochs']
    steps_per_epoch = config['steps_per_epoch']
    
    for epoch in range(1, epochs + 1):
        gen_model.train()
        dis_model.train()
        
        total_d_loss = 0
        total_g_loss = 0
        
        for step in tqdm(range(steps_per_epoch), desc = f'Epoch {epoch}/{epochs}: ', ncols = 100):
            lr_images, gt_images = next(iterator)
            batch_size = lr_images.shape[0]
            lr_images = lr_images.to(device)
            gt_images = gt_images.to(device)
            
            # Train generator    
            sr_images = gen_model(lr_images)
            sr_out = dis_model(sr_images.detach())
                    
            real_label = Uniform(0.96, 1.0).sample([batch_size, 1]).to(device)
    #         real_label = torch.ones(batch_size, 1).to(device)
            pixel_loss = pixel_criterion(sr_images, gt_images)
            content_loss = content_criterion(sr_images, gt_images)
            adversarial_loss = adversarial_criterion(sr_out, real_label)
            
    #         g_loss = pixel_loss + content_loss + adversarial_weight*adversarial_loss
            g_loss = content_loss + adversarial_weight*torch.sum(-torch.log(sr_out))
            
            gen_optim.zero_grad()
            g_loss.backward()
            gen_optim.step()
            
            ## Train Discriminator
            gt_out = dis_model(gt_images.detach())
            d_gt_loss = adversarial_criterion(gt_out, real_label)
            
            dis_optim.zero_grad()
            d_gt_loss.backward()
            dis_optim.step()
            
            sr_out = dis_model(sr_images.detach())
            fake_label = Uniform(0., 0.05).sample([batch_size, 1]).to(device)
    #         fake_label = torch.zeros(batch_size, 1).to(device)

            d_sr_loss = adversarial_criterion(sr_out, fake_label)
            
            dis_optim.zero_grad()
            d_sr_loss.backward()
            dis_optim.step()
            
            d_loss = d_gt_loss + d_sr_loss
            
            total_d_loss += d_loss
            total_g_loss += g_loss
            
            if count_steps == num_steps:
                iterator = iter(train_loader)
                count_steps = 0
            count_steps += 1

        average_d_loss = total_d_loss.item()/steps_per_epoch
        average_g_loss = total_g_loss.item()/steps_per_epoch
        
        print(f'Epoch: {epoch}/{epochs}: D Loss: {average_d_loss : .4f}, G loss: {average_g_loss : .4f}')

    return gen_model, dis_model

        # for jupyter notebook
        # if epoch % 1 == 0:
        #     with torch.no_grad():
        #         sr_images = gen_model(fixed_input.to(device)).detach().cpu()
        #     results = torch.cat([sr_images, fixed_output], dim = 0)
        #     results = make_grid(results, padding=2, normalize=True)
        #     img_list.append(results)
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(img_list[-1].permute(1,2,0))
        #     plt.show()
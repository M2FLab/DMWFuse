import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import TrainDataset
from net.DMWFuse import DMW_Fuse
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.loss_utils import *

class DMWFuse(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DMW_Fuse()
        self.loss_fn  = nn.L1Loss(reduction='mean')  
        self.FusionLoss = Fusionloss()
        self.WaveletLoss = nn.L1Loss(reduction='mean')

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        print(f"当前批次图片名称: clean_name={clean_name[0]}, 退化类型ID={de_id[0]}")
        restored, loss_b = self.net(degrad_patch)

        L1_V = self.loss_fn(restored,clean_patch)
        loss_grad, loss_laplacian = self.FusionLoss(clean_patch, restored)
        
        _, high_freq_clean = self.net.wavelet_extractor(clean_patch)
        _, high_freq_restored = self.net.wavelet_extractor(restored)
        wavelet_loss = self.WaveletLoss(high_freq_restored, high_freq_clean)
        
        loss = L1_V.cuda('cuda:0') + 10 * loss_b.cuda('cuda:0') + 10 * loss_grad.cuda('cuda:0') + 5 * wavelet_loss.cuda('cuda:0')
        print("train_loss={:.4f}, L1_V={:.4f}, loss_b={:.4f}, loss_grad={:.4f}, wavelet_loss={:.4f}".format(loss, L1_V, loss_b, loss_grad, wavelet_loss))
        self.log("train_loss", loss)
        self.log("L1", L1_V)
        self.log("loss_b", loss_b)
        self.log("loss_grad", loss_grad)
        self.log("wavelet_loss", wavelet_loss)
        
        return loss
    
        
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=5,max_epochs=180)
        return [optimizer],[scheduler]

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")
    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_path, every_n_epochs = 5, save_top_k=-1)
    trainloader = DataLoader(trainset,
                             batch_size=opt.batch_size,
                             pin_memory=True,
                             shuffle=True,
                             drop_last=True,
                             num_workers=opt.num_workers)

    model = DMWFuse()
    trainer = pl.Trainer( max_epochs=opt.epochs,
                          accelerator="gpu",
                          devices=opt.num_gpus,
                          strategy="ddp_find_unused_parameters_true",
                          logger=logger,
                          callbacks=[checkpoint_callback]
                          )
    if opt.use_ckpt:
        trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path = opt.ckpt_dir)
    else:
        trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    import os
    os.environ["WANDB_DISABLED"] = "true"
    main()
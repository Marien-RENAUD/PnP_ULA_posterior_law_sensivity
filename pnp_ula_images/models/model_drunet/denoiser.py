import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
from .dpir_unet import DPIRNNclass #,PotentialNNclass,REDPotentialNNclass,GSPNPNNclass
from torchmetrics import PeakSignalNoiseRatio as PSNR
from random import uniform,choice
from os.path import join
from os import listdir
from PIL.Image import open as imopen
import torchvision
from torchvision.transforms.functional import to_tensor,rgb_to_grayscale
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from torch.optim import Adam
from torch.optim import lr_scheduler
from torchvision.transforms.functional import center_crop

class Denoiser(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model=DPIRNNclass(numInChan=self.hparams.numInChan,numOutChan=self.hparams.numOutChan,network=self.hparams.denoiser_name,train_network=True,sigma_map=self.hparams.sigma_map)
        self.train_PSNR=PSNR(data_range=1.0)
        self.val_psnr_module=nn.ModuleDict({f'val_psnr_{i}':PSNR(data_range=1.0) for i in range(len(self.hparams.sigma_test_list))})
        if hparams.loss == 'mse':
            self.lossFunc= torch.nn.MSELoss()
        elif hparams.loss == 'l1':
            self.lossFunc= torch.nn.L1Loss()
        if self.hparams.task=='color':
            self.testNames=['butterfly.png','leaves.png','starfish.png']
            testLst=[to_tensor(imopen(join('../miscs','set3c',n)).convert('RGB')) for n in self.testNames]
            self.testTensor=torch.stack(testLst,dim=0)
        elif self.hparams.task=='mri':
            self.testNames=listdir(join(self.hparams.dataset_path,'BrainMRI256/BrainImages_test'))[:3]
            testLst=[to_tensor(imopen(join(self.hparams.dataset_path,'BrainMRI256/BrainImages_test',n))) for n in self.testNames]
            self.testTensor=torch.stack(testLst,dim=0)
        elif self.hparams.task=='gray':
            self.testNames=['butterfly.png','leaves.png','starfish.png']
            testLst=[to_tensor(imopen(join('../miscs','set3c',n)).convert('RGB')) for n in self.testNames]
            self.testTensor=torch.stack(testLst,dim=0)
            self.testTensor=rgb_to_grayscale(self.testTensor)

    def forward(self, x,sigma,create_graph=True,strict=True):
        return self.model(x,sigma,create_graph=create_graph,strict=strict)

    def training_step(self, batch, batch_idx):
        gtImg,_ = batch
        if self.hparams.fixed_sigma:
            sigma=self.hparams.sigma_min/255.0
        else:
            sigma= uniform(self.hparams.sigma_min,self.hparams.sigma_max)/255.0
        noise=torch.randn_like(gtImg)*sigma
        noisyImg=gtImg+noise
        predImg=self(noisyImg,sigma,create_graph=True,strict=True)
        if self.hparams.train_noise:
            loss=self.lossFunc(noise,noisyImg-predImg)
        else:
            loss=self.lossFunc(gtImg,predImg)
        self.log('train_loss',loss.detach(), prog_bar=False,on_step=True,logger=True)
        self.train_PSNR.update(gtImg,predImg)
        psnr=self.train_PSNR.compute().detach()
        self.log('train_PSNR',psnr, prog_bar=True,on_step=True,logger=True)
        self.train_PSNR.reset()
        return {'loss':loss,'psnr':psnr}

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(False)
        with torch.no_grad():
            gtImgs,_ = batch
            sigma_list = self.hparams.sigma_test_list
            for (name, psnr_calculator), sigma  in zip(self.val_psnr_module.items(),sigma_list):
                sigma=sigma/255.0
                noise=torch.randn_like(gtImgs)*sigma
                noisyImgs=gtImgs+noise
                predImgs=self(noisyImgs,sigma,create_graph=False,strict=False)
                predNoise=gtImgs-predImgs
                loss=self.lossFunc(predNoise,noise)
                psnr_calculator.update(gtImgs,predImgs)
                self.log('val_loss_sigma:%d'%sigma,loss.detach(), prog_bar=False,on_step=False,logger=True)
        return {'val_loss':loss}
    def validation_epoch_end(self, outputs):
        torch.set_grad_enabled(False)
        with torch.no_grad():
            sigma_list = self.hparams.sigma_test_list
            val_total_psnr=0
            for name, psnr_calculator in self.val_psnr_module.items():
                psnr=psnr_calculator.compute()
                self.log(name, psnr.detach(), prog_bar=False,logger=True, sync_dist=True)
                psnr_calculator.reset()
                val_total_psnr+=psnr
            val_total_psnr/=len(sigma_list)
            self.log('val_total_psnr', val_total_psnr.detach(), prog_bar=True,logger=True, sync_dist=True)
            testTensor=self.testTensor.to(self.device)
            for i in range(len(self.hparams.sigma_test_list)):
                sigma=self.hparams.sigma_test_list[i]/255.0
                noise=torch.randn_like(testTensor)*sigma
                noisyImgs=testTensor+noise
                predImgs=self(noisyImgs,sigma,create_graph=False,strict=False)
                for j in range(len(testTensor)):
                    gtImg=testTensor[j]
                    predImg=predImgs[j]
                    outpsnr=cal_psnr(gtImg.detach().cpu().numpy().transpose(1,2,0),predImg.detach().cpu().numpy().transpose(1,2,0),data_range=1.0)
                    self.log(f'test_PSNR_sigma:{sigma},img_{self.testNames[j]}',outpsnr, prog_bar=False,on_step=False,logger=True, sync_dist=True)
                clean_grid = torchvision.utils.make_grid(testTensor.detach(),normalize=True,nrow=2)
                noisy_grid = torchvision.utils.make_grid(noisyImgs.detach(),normalize=True,nrow=2)
                recon_grid = torchvision.utils.make_grid(torch.clamp(predImgs,min=0.0,max=1.0).detach(),normalize=False,nrow=2)
                self.logger.experiment.add_image(f'test_image/clean/sigma-{sigma}', clean_grid, self.current_epoch)
                self.logger.experiment.add_image(f'test_image/noisy/sigma-{sigma}', noisy_grid, self.current_epoch)
                self.logger.experiment.add_image(f'test_image/recon/sigma-{sigma}', recon_grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.optimizer_lr,weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
    @staticmethod
    def add_denoiser_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--denoiser_name', type=str, default='dncnn')

        parser.add_argument('--task', type=str, choices=['color','mri','gray'], default='gray')
        parser.add_argument('--numInChan', type=int, default=3)
        parser.add_argument('--numOutChan', type=int, default=3)
        parser.add_argument('--sigma_min', type=float, default=2.55)
        parser.add_argument('--sigma_max', type=float, default=12.75)
        parser.add_argument('--fixed_sigma', dest='fixed_sigma', action='store_true')
        parser.set_defaults(fixed_sigma=False)
        parser.add_argument('--sigma_test_list', type=float, nargs='+', default=[2.55,7.5,12.75])
        parser.add_argument('--loss', type=str, default='mse')
        parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_denoiser', type=str, default='')
        parser.add_argument('--enable_pretrained_denosier',dest='enable_pretrained_denosier',action='store_true')
        parser.set_defaults(enable_pretrained_denosier=False)
        parser.add_argument('--pretrained_checkpoint', type=str,default='')
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[300, 600, 900, 1200])
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
        parser.add_argument('--optimizer_lr', type=float, default=1e-4)

        parser.add_argument('--no_sigma_map', dest='sigma_map', action='store_false')
        parser.set_defaults(sigma_map=True)
        parser.add_argument('--train_noise', dest='train_noise', action='store_true')
        parser.set_defaults(train_noise=False)

        return parser

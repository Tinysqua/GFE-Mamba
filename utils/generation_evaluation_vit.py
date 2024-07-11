import glob
import os
import sys; sys.path.append('./')
from utils.common import see_mri_pet
import torch
from dataloader.threed_loader import form_dataloader
from dataloader.pic_table_loader import classi_dataloader
import argparse
from os.path import join as j
from utils.common import load_config
from pytorch3dunet.unet3d.model import Residual_mid_UNet3D_vit
from torchvision.utils import save_image
from tqdm import tqdm

def main(args):
    device = 'cuda'
    cf = load_config(glob.glob(j(args.model_updir, "*.yaml"))[0])
    eval_path = '/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/ADNI_1year_test'
    # test_loader = form_dataloader(eval_path, 
    #                                 cf['img_sz'], 
    #                                     1, 
    #                                     True)
    test_loader = classi_dataloader(eval_path, 
                                    cf['img_sz'], 
                                        1, 
                                        '/home/chenyifei/data/CYFData/ZSH/Datasets/ADNI_1&3year/filted_2&5_1year_delcol.csv', 
                                        False)
    model = Residual_mid_UNet3D_vit(1, 1, is_segmentation=False, f_maps=(64, 128, 256))
    # j(args.model_updir, 'model_save', 'model.pt')
    model.load_state_dict(torch.load(j(args.model_updir, 'model_save', 'model.pt')))
    model = model.to(device).eval()
    save_path = j(args.model_updir, "eval_save")
    os.makedirs(save_path, exist_ok=True)
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        if (i+1) == 20:
            break
        condition, _ = batch['image'], batch['label']
        condition = condition.to(device)
        with torch.no_grad():
            reconstruction = model(condition)
        images = torch.cat([condition, reconstruction], dim=-2)
        save_image(see_mri_pet(images), j(save_path, f'{i+1}.png'))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_updir', type=str, default='weights/exp_0327124326_main_gan_vit')
    args = parser.parse_args()
    main(args)

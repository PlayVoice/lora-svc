import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from omegaconf import OmegaConf

from model.generator import Generator
from model.discriminator import Discriminator


def main(args):
    hp = OmegaConf.load(args.config)

    model_g = Generator(hp)
    model_d = Discriminator(hp)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

    model_g.load_state_dict(checkpoint['model_g'])
    model_d.load_state_dict(checkpoint['model_d'])

    torch.save({
        'model_g': model_g.state_dict(),
    }, "maxgan_g.pth")

    torch.save({
        'model_g': model_g.state_dict(),
        'model_d': model_d.state_dict(),
    }, "maxgan_pretrain_32K.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    args = parser.parse_args()

    main(args)

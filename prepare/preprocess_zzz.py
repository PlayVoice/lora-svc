import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    args = parser.parse_args()
    hp = OmegaConf.load(args.config)

    trainloader = create_dataloader(hp, True)
    for _ in tqdm(trainloader):
        pass

    valloader = create_dataloader(hp, False)
    for _ in tqdm(trainloader):
        pass

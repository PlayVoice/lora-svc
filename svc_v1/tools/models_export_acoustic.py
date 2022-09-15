import torch
import utils
from singer_vc.models import SynthesizerTrn


# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).cuda()

_ = utils.load_checkpoint("./logs/svc/G_50000.pth", net_g, None)
net_g.eval()
net_g.remove_weight_norm()
print("==========init ok==========")
#
torch.save(net_g, "model_net_g.pth")
print("==========save ok==========")
#
net_g = torch.load("model_net_g.pth")
print("==========load ok==========")

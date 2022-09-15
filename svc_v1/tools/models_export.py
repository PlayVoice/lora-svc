import torch
import utils
from singer_vits_mel.models import SynthesizerTrn
from pitch.pit_models import PitchExtractor


# define model and load checkpoint
hps = utils.get_hparams_from_file("./configs/singing_base.json")

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).cuda()

_ = utils.load_checkpoint("./logs/singer_vits_mel/G_230000.pth", net_g, None)
net_g.eval()
net_g.remove_weight_norm()

net_p = PitchExtractor().cuda()
_ = utils.load_checkpoint("./logs/singing_pitch/P_500000.pth", net_p, None)
net_p.eval()
print("==========init ok==========")
#
torch.save(net_g, "model_net_g.pth")
torch.save(net_p, "model_net_p.pth")
print("==========save ok==========")
#
net_g = torch.load("model_net_g.pth")
net_p = torch.load("model_net_p.pth")
print("==========load ok==========")

import json
from pathlib import Path
from PIL import Image
from itertools import chain

import torch
from sconf import Config
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
from base.dataset import read_font, render
from base.utils import save_tensor_to_image, load_reference
from LF.models import Generator
from inference import infer_LF
###############################################################
weight_path = "temp/outputs2/checkpoints/last.pth"  # path to weight to infer
emb_dim = 8
decomposition = "data/kor/decomposition.json"
primals = "data/kor/primals.json"
###############################################################

decomposition = json.load(open(decomposition))
primals = json.load(open(primals))
n_comps = len(primals)

def decompose_to_ids(char):
    dec = decomposition[char]
    comp_ids = [primals.index(d) for d in dec]
    return comp_ids

cfg = Config("cfgs/LF/p2/default.yaml")

gen = Generator(n_comps=n_comps, emb_dim=emb_dim).cuda().eval()
weight = torch.load(weight_path)
if "generator_ema" in weight:
    weight = weight["generator_ema"]
gen.load_state_dict(weight)

########################################################
ref_path = "data_example/kor/png"
extension = "png"
ref_chars = "값넋닻볘츄퀭핥훟"
## Comment upper lines and uncomment lower lines to test with ttf files.
# extension = "ttf"
# ref_chars = "값같곬곶깎넋늪닫닭닻됩뗌략몃밟볘뺐뽈솩쐐앉않얘얾엌옳읊죡쮜춰츄퀭틔핀핥훟"
########################################################

ref_dict, load_img = load_reference(ref_path, extension, ref_chars)

########################################################
gen_chars = "좋은하루되세요"  # Characters to generate
save_dir = "./result/lf"  # Directory where you want to save generated images
source_path = "data/kor/source.ttf"
source_ext = "ttf"
batch_size = 16
########################################################

infer_LF(gen, save_dir, source_path, source_ext, gen_chars, ref_dict, load_img,
         decomposition, primals, batch_size)

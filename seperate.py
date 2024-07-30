import os
import yaml
import torch as th
import torch.nn as nn
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm

from loader.datareader_one import DataReader
from model.e3net import E3Net as model
# from model.e3net_addblock import E3Net as model

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def run(args):
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    data_reader = DataReader(**conf["datareader"])
    device = th.device("cuda" if conf["test"]["use_cuda"] and th.cuda.is_available() else "cpu")

    nnet = model(**conf["nnet_conf"])

    checkpoint_dir = Path(conf["test"]["checkpoint"])
    cpt_fname = checkpoint_dir / "best.pt.tar"
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.reload_spk()

    nnet.load_state_dict(cpt["model_state_dict"])
    nnet = nnet.to(device)
    nnet.eval()

    if not os.path.exists(conf["save"]["dir"]):
        os.makedirs(conf["save"]["dir"])

    with th.no_grad():
        for egs in tqdm(data_reader):
            egs = load_obj(egs, device)
            egs["mix"] = egs["mix"].contiguous()
            egs["aux"] = egs["aux"].contiguous()
            est = nn.parallel.data_parallel(nnet, (egs["mix"], egs["aux"]))

            out = est["wav"].detach().squeeze().cpu().numpy()
            out = out / np.max(np.abs(out)) * egs["max_norm"]
            sf.write(os.path.join(conf["save"]["dir"], egs["utt_id"]), out, conf["save"]["sample_rate"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Command to test separation model in Pytorch",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    args = parser.parse_args()
    run(args)

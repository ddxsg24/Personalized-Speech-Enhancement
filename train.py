import pprint
import argparse
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import torch as th
import numpy as np
from pathlib import Path

from loader.dataloader import make_auto_loader

from trainer.trainer import Trainer

from model.e3net import E3Net as model


def make_optimizer(params, opt):
    '''
    make optimizer
    '''
    supported_optimizer = {
        "sgd": th.optim.SGD,  # momentum, weight_decay, lr
        "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
        "adam": th.optim.Adam,  # weight_decay, lr
        "adadelta": th.optim.Adadelta,  # weight_decay, lr
        "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
        "adamax": th.optim.Adamax  # lr, weight
        # ...
    }

    if opt['optim']['name'] not in supported_optimizer:
        raise ValueError("Now only support optimizer {}".format(opt['optim']['name']))
    optimizer = supported_optimizer[opt['optim']['name']](params, **opt['optim']['optimizer_kwargs'])
    return optimizer

def make_dataloader(opt):
    # make train's dataloader
    train_dataloader = make_auto_loader(
        opt['datasets']['train']['clean_scp'],
        opt['datasets']['train']['clean_spk'],
        opt['datasets']['train']['infer_scp'],
        opt['datasets']['train']['noise_scp'],
        opt['datasets']['train']['rir_scp'],
        **opt['datasets']['dataloader_setting'])

    # make validation dataloader
    valid_dataloader = make_auto_loader(
        opt['datasets']['val']['clean_scp'],
        opt['datasets']['val']['clean_spk'],
        opt['datasets']['val']['infer_scp'],
        opt['datasets']['val']['noise_scp'],
        opt['datasets']['val']['rir_scp'],
        **opt['datasets']['dataloader_setting'])
    return train_dataloader, valid_dataloader

def run(args):

    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))), flush=True)

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)

    checkpoint_dir = Path(conf['train']['checkpoint'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    th.cuda.manual_seed_all(conf['train']['seed'])

    # if exist, resume training
    last_checkpoint = checkpoint_dir / "last.pt.tar"
    if last_checkpoint.exists():
        print(f"Found old checkpoint: {last_checkpoint}", flush=True)
        conf['train']['resume'] = last_checkpoint.as_posix()

    # dump configurations
    with open(checkpoint_dir / "train.yaml", "w") as f:
        yaml.dump(conf, f)
    
    #build nnet
    nnet = model(**conf["nnet_conf"])
    # build optimizer
    optimizer = make_optimizer(nnet.parameters(), conf)
    # build dataloader
    train_loader, valid_loader = make_dataloader(conf)
    '''
    需要修改
    '''
    # build scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = conf['scheduler']['T_max'],
        eta_min = conf['scheduler']['eta_min'],
        last_epoch= conf['scheduler']['last_epoch'])

    device = th.device('cuda' if conf['train']['use_cuda'] and th.cuda.is_available() else 'cpu')

    trainer = Trainer(nnet,
                      optimizer,
                      scheduler,
                      device,
                      conf)

    if conf['train']['eval_interval'] > 0:
        trainer.run_batch_per_epoch(train_loader,
                                    valid_loader,
                                    num_epochee=conf['train']['epoch'],
                                    eval_interval=conf['train']['eval_interval'])
    else:
        trainer.run(train_loader,
                    valid_loader,
                    num_epoches=conf['train']['epoch'],
                    test=conf['test'],
                   )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to train separation model in Pytorch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    args = parser.parse_args()
    run(args)
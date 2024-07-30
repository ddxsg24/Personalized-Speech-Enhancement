import os
import sys
import time

from pathlib import Path
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import os
# print(os.path.dirname(__file__))

from loss.loss import sisnr_loss, spk_loss, mse_loss, mae_loss

sys.path.append(
    os.path.dirname(__file__))
from logger.logger import get_logger

# from torch import autograd
# th.autograd.set_detect_anomaly(True)

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

class SimpleTimer(object):
    '''
    A simple timer
    '''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60

class ProgressReporter(object):
    '''
    A sample progress reporter
    '''
    def __init__(self, logger, period=100):
        self.period = period
        if isinstance(logger, str):
            self.logger = get_logger(logger, file=True)
        else:
            self.logger = logger
        self.header = "Trainer"
        self.reset()
    
    def log(self, sstr):
        self.logger.info(f"{self.header}: {sstr}")
    
    def eval(self):
        self.log("set eval mode...")
        self.mode = "eval"
        self.reset()
    
    def train(self):
        self.log("set train mode...")
        self.mode = "train"
        self.reset()
    
    def reset(self):
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def add(self, key, value, batch_num, epoch):
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            avg = sum(self.stats[key][-self.period:]) / self.period
            self.log(f"Epoch:{epoch} processed {N:.2e} / {batch_num:.2e} batches ({key} = {avg:+.2f})...")
        
    def report(self, epoch, lr):
        N = len(self.stats["loss"])
        if self.mode == "eval":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"loss on {N:d} batches: {sstr}")
        
        loss = sum(self.stats["loss"]) / N
        cost = self.timer.elapsed()
        sstr = f"Loss(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: " + f"{self.mode} = {loss:.4f}({cost:.2f}m/{N:d})"
        return loss, sstr

class Trainer(object):
    '''
    Basic neural network trainer
    '''
    def __init__(self,
                 nnet,
                 optimizer,
                 scheduler,
                 device,
                 conf):

        self.default_device = device

        self.checkpoint = Path(conf['train']['checkpoint'])
        self.checkpoint.mkdir(exist_ok=True, parents=True)
        self.reporter = ProgressReporter(
            (self.checkpoint / "trainer.log").as_posix() if conf['logger']['path'] is None else conf['logger']['path'],
            period=conf['logger']['print_freq'])
        
        self.gradient_clip = conf['optim']['gradient_clip']
        self.start_epoch = 0 # zero based
        self.no_impr = conf['train']['early_stop']
        self.save_period = conf['train']['save_period']

        # only network part
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6
        
        # logging
        self.reporter.log("model summary:\n{}".format(nnet))
        self.reporter.log(f"#param: {self.num_params:.2f}M")
        
        if conf['train']['resume']:
            # resume nnet and optimizer from checkpoint
            if not Path(conf['train']['resume']).exists():
                raise FileNotFoundError(
                    f"Could not find resume checkpoint: {conf['train']['resume']}")
            cpt = th.load(conf['train']['resume'], map_location="cpu")
            self.start_epoch = cpt["epoch"]
            self.reporter.log(
                f"resume from checkpoint {conf['train']['resume']}: epoch {self.start_epoch:d}")
            # load nnet
            nnet.reload_spk(conf['train']['spk_resume'])
            nnet.load_state_dict(cpt["model_state_dict"], strict=True)
            self.nnet = nnet.to(self.default_device)

            optimizer.load_state_dict(cpt["optim_state_dict"])
            self.optimizer = optimizer
        elif conf['train']['spk_resume']:
            self.reporter.log(
                f"resume spk model from checkpoint {conf['train']['spk_resume']}")
            nnet.reload_spk(conf['train']['spk_resume'])
            self.nnet = nnet.to(self.default_device)
            self.optimizer = optimizer
        else:
            self.nnet = nnet.to(self.default_device)
            self.optimizer = optimizer
        
        if conf['optim']['gradient_clip']:
            self.reporter.log(
                f"gradient clipping by {conf['optim']['gradient_clip']}, default L2")
            self.clip_norm = conf['optim']['gradient_clip']
        else:
            self.clip_norm = 0
        
        self.scheduler = scheduler

    def save_checkpoint(self, epoch, best=True):
        '''
        Save checkpoint (epoch, model, optimizer)
        '''
        cpt = {
            "epoch": epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        cpt_name = "{0}.pt.tar".format("best" if best else "last")
        th.save(cpt, self.checkpoint / cpt_name)
        self.reporter.log(f"save checkpoint {cpt_name}")
        if self.save_period > 0 and epoch % self.save_period == 0:
            th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")

    def train(self, data_loader, epoch):
        self.nnet.train()
        self.nnet.spk_model.eval()
        self.reporter.train()
        batch_num = len(data_loader) # data_loader是根据参数做好的一个batch的数据，包括训练一个网络需要的所有输入

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.default_device)
            # contiguous跟数据分布有关，建议后面看看
            egs["mix"] = egs["mix"].contiguous()
            egs["ref"] = egs['ref'].contiguous()
            egs["aux"] = egs["aux"].contiguous()
            egs["spk_label"] = egs["spk_label"].contiguous()
            self.optimizer.zero_grad()
            # 跟并行计算有关的
            est = nn.parallel.data_parallel(self.nnet, (egs["mix"], egs["aux"]))

            snr_loss = sisnr_loss(est["wav"], egs["ref"])
            ce_loss = spk_loss(est["spk_pred"], egs["spk_label"])
            l1_loss = mae_loss(est["wav"], egs["ref"])
            # phase_loss = get_phasen_loss(est["wav"], egs["ref"])
            # cplx_mse_loss = RI_Mag_Compress_Mse_Asym(est["wav"], egs["ref"]) * 0.1

            loss = 1.0 * snr_loss + 0.5 * ce_loss + 1.0 * l1_loss

            # with autograd.detect_anomaly():
            loss.backward()

            self.reporter.add("snr_loss", snr_loss.item(), batch_num, epoch)
            # self.reporter.add("cplx_mse_loss", cplx_mse_loss.item(), batch_num, epoch)
            # self.reporter.add("phase_loss", phase_loss.item(), batch_num, epoch)
            self.reporter.add("ce_loss", ce_loss.item(), batch_num, epoch)
            self.reporter.add("mae_loss", l1_loss.item(), batch_num, epoch)
            self.reporter.add("loss", loss.item(), batch_num, epoch)

            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)
                self.reporter.add("norm", norm, batch_num, epoch)
            self.optimizer.step()
    
    def eval(self, data_loader, epoch):
        self.nnet.eval()
        self.reporter.eval()
        batch_num = len(data_loader)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                egs["mix"] = egs["mix"].contiguous()
                egs["ref"] = egs["ref"].contiguous()
                egs["aux"] = egs["aux"].contiguous()
                egs["spk_label"] = egs["spk_label"].contiguous()

                est = nn.parallel.data_parallel(self.nnet, (egs["mix"], egs["aux"]))

                snr_loss = sisnr_loss(est["wav"], egs["ref"])
                ce_loss = spk_loss(est["spk_pred"], egs["spk_label"])
                l1_loss = mae_loss(est["wav"], egs["ref"])
                # phase_loss = get_phasen_loss(est["wav"], egs["ref"])
                # cplx_mse_loss = RI_Mag_Compress_Mse_Asym(est["wav"], egs["ref"]) * 0.1

                loss = 1.0 * snr_loss + 0.5 * ce_loss + 1.0 * l1_loss
                
                self.reporter.add("snr_loss", snr_loss.item(), batch_num, epoch)
                # self.reporter.add("cplx_mse_loss", cplx_mse_loss.item(), batch_num, epoch)
                # self.reporter.add("phase_loss", phase_loss.item(), batch_num, epoch)
                self.reporter.add("ce_loss", ce_loss.item(), batch_num, epoch)
                self.reporter.add("mae_loss", l1_loss.item(), batch_num, epoch)
                self.reporter.add("loss", loss.item(), batch_num, epoch)
    
    def run(self, train_loader, valid_loader, num_epoches=50, test=False):
        '''
        Run on whole training set and evaluate
        '''
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # avoid alloc memory grom gpu0
        # th.cuda.set_device(self.default_device)

        if test:
            e = self.start_epoch
            # make sure not inf
            best_loss = 10000
            self.scheduler.best = 10000
            no_impr = 0
        else:
            e = self.start_epoch
            self.eval(valid_loader, e)
            best_loss, _ = self.reporter.report(e, 0)
            self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
            # make sure not inf
            self.scheduler.best = best_loss
            no_impr = 0

        while e < num_epoches:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]

            # >> train
            self.train(train_loader, e)
            _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(valid_loader, e)
            cv_loss, sstr = self.reporter.report(e, cur_lr)
            if cv_loss > best_loss:
                no_impr += 1
                sstr += f"| no impr, best = {self.scheduler.best:.4f}"
            else:
                best_loss = cv_loss
                no_impr = 0
                self.save_checkpoint(e, best=True)
            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(cv_loss)
            # flush scheduler info
            sys.stdout.flush()
            # save checkpoint
            self.save_checkpoint(e, best=False)
            if no_impr == self.no_impr:
                self.reporter.log(
                    f"stop training cause no impr for {no_impr:d} epochs")
                break
        self.reporter.log(f"training for {e:d}/{num_epoches:d} epoches done!")

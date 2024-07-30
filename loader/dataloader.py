import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random

import torch
import torch as th

import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp 

eps = np.finfo(np.float32).eps

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def clip_data(data, start, segment_length):
    tgt = np.zeros(segment_length)
    data_len = data.shape[0]
    if start == -2:
        """
        this means segment_length // 4 < data_len < segment_length // 2
        padding to A_A_A
        """
        if data_len < segment_length//3:
            data = np.pad(data, [0, segment_length//3 - data_len])
            tgt[:segment_length//3] += data
            st = segment_length//3
            tgt[st:st+data.shape[0]] += data
            st = segment_length//3 * 2
            tgt[st:st+data.shape[0]] += data
        
        else:
            """
            padding to A_A
            """
            data = np.pad(data, [0, segment_length//2 - data_len])
            tgt[:segment_length//2] += data
            st = segment_length//2
            tgt[st:st+data.shape[0]] += data
    
    elif start == -1:
        '''
        this means segment_length < data_len*2
        padding to A_A
        '''
        if data_len % 4 == 0:
            tgt[:data_len] += data
            tgt[data_len:] += data[:segment_length-data_len]
        elif data_len % 4 == 1:
            tgt[:data_len] += data
        elif data_len % 4 == 2:
            tgt[-data_len:] += data
        elif data_len % 4 == 3:
            tgt[(segment_length-data_len)//2:(segment_length-data_len)//2+data_len] += data
    
    else:
        tgt += data[start:start+segment_length]
    
    return tgt


def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e * (10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy >= low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
    new_noise = noise * k
    return new_noise

def mix_speech(clean, infer, snr, randstate):
    clean_length = clean.shape[0]
    infer_length = infer.shape[0]
    if clean_length > infer_length:
        st = randstate.randint(clean_length + 1 - infer_length)
        infer_t = np.zeros(clean_length)
        infer_t[st:st+infer_length] = infer
        infer = infer_t
    elif clean_length < infer_length:
        st = randstate.randint(infer_length + 1 - clean_length)
        infer = infer[st:st+clean_length]
    
    snr_infer = snr_mix(clean, infer, snr)
    return snr_infer

def mix_noise(clean, noise, snr, randstate):
    clean_length = clean.shape[0]
    noise_length = noise.shape[0]
    if clean_length > noise_length:
        st = randstate.randint(clean_length + 1 - noise_length)
        noise_t = np.zeros(clean_length)
        noise_t[st:st+noise_length] = noise
        noise = noise_t
    elif clean_length < noise_length:
        st = randstate.randint(noise_length + 1 - clean_length)
        noise = noise[st:st+clean_length]
    
    snr_noise = snr_mix(clean, noise, snr)
    return snr_noise

def add_reverb(cln_wav, rir_wav):
    # cln_wav: L
    # rir_wav: L
    wav_tgt = sps.oaconvolve(cln_wav, rir_wav)
    wav_tgt = wav_tgt[:cln_wav.shape[0]]
    return wav_tgt

def get_one_spks_noise(clean, noise, randstate):
    snr_se = randstate.uniform(-5, 20)
    gen_noise = mix_noise(clean, noise, snr_se, randstate)
    noisy = clean + gen_noise

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def get_two_spks_noise(clean, infer1, noise, randstate):
    snr_se = randstate.uniform(-5, 20)
    gen_noise = mix_noise(clean, noise, snr_se, randstate)
    snr_ss1 = randstate.uniform(-5, 20)
    gen_infer1 = mix_speech(clean, infer1, snr_ss1, randstate)
    noisy = clean + gen_infer1 + gen_noise

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def get_three_spks_noise(clean, infer1, infer2, noise, randstate):
    snr_se = randstate.uniform(-5, 20)
    gen_noise = mix_noise(clean, noise, snr_se, randstate)
    snr_ss1 = randstate.uniform(-5, 20)
    gen_infer1 = mix_speech(clean, infer1, snr_ss1, randstate)
    snr_ss2 = randstate.uniform(-5, 20)
    gen_infer2 = mix_speech(clean, infer2, snr_ss2, randstate)
    noisy = clean + gen_infer1 + gen_infer2 + gen_noise

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def get_two_spks(clean, infer1, randstate):
    snr_ss1 = randstate.uniform(-5, 20)
    gen_infer1 = mix_speech(clean, infer1, snr_ss1, randstate)
    noisy = clean + gen_infer1

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def get_three_spks(clean, infer1, infer2, randstate):
    snr_ss1 = randstate.uniform(-5, 20)
    gen_infer1 = mix_speech(clean, infer1, snr_ss1, randstate)
    snr_ss2 = randstate.uniform(-5, 20)
    gen_infer2 = mix_speech(clean, infer2, snr_ss2, randstate)
    noisy = clean + gen_infer1 + gen_infer2

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

def generate_data(clean_path, infer1_path, infer2_path, noise_path, clean_rir_path, infer1_rir_path,
                    infer2_rir_path, noise_rir_path, start, segment_length, randstate):
    
    clean = get_firstchannel_read(clean_path)
    clean = clip_data(clean, start, segment_length)
    choice_rir = randstate.uniform(0, 100)
    # add rir
    if choice_rir < 50:
        clean_rir = get_firstchannel_read(clean_rir_path)
        clean = add_reverb(clean, clean_rir)

    # choice_senario = 10
    # choice_senario = 30
    # choice_senario = 50
    # choice_senario = 70
    # choice_senario = 90
    choice_senario = randstate.uniform(0, 100)
    # [0, 20]: +noise
    # [20, 40]: +infer
    # [40, 60]: +infer +infer
    # [60, 80]: +infer +noise
    # [80, 100]: +infer +infer +noise

    # +noise
    if choice_senario < 20:
        noise = get_firstchannel_read(noise_path)
        if choice_rir < 50:
            noise_rir = get_firstchannel_read(noise_rir_path)
            noise = add_reverb(noise, noise_rir) 
        inputs, labels = get_one_spks_noise(clean, noise, randstate)
    
    # +infer
    elif choice_senario < 40:
        infer1 = get_firstchannel_read(infer1_path)
        if choice_rir < 50:
            infer1_rir = get_firstchannel_read(infer1_rir_path)
            infer1 = add_reverb(infer1, infer1_rir)
        inputs, labels = get_two_spks(clean, infer1, randstate)
    
    # +infer +infer
    elif choice_senario < 60:
        infer1 = get_firstchannel_read(infer1_path)
        if choice_rir < 50:
            infer1_rir = get_firstchannel_read(infer1_rir_path)
            infer1 = add_reverb(infer1, infer1_rir)
        infer2 = get_firstchannel_read(infer2_path)
        if choice_rir < 50:
            infer2_rir = get_firstchannel_read(infer2_rir_path)
            infer2 = add_reverb(infer2, infer2_rir)
        inputs, labels = get_three_spks(clean, infer1, infer2, randstate)

    # +infer +noise
    elif choice_senario < 80:
        noise = get_firstchannel_read(noise_path)
        if choice_rir < 50:
            noise_rir = get_firstchannel_read(noise_rir_path)
            noise = add_reverb(noise, noise_rir) 
        infer1 = get_firstchannel_read(infer1_path)
        if choice_rir < 50:
            infer1_rir = get_firstchannel_read(infer1_rir_path)
            infer1 = add_reverb(infer1, infer1_rir)
        inputs, labels = get_two_spks_noise(clean, infer1, noise, randstate)

    # +infer +infer +noise
    elif choice_senario < 100:
        noise = get_firstchannel_read(noise_path)
        if choice_rir < 50:
            noise_rir = get_firstchannel_read(noise_rir_path)
            noise = add_reverb(noise, noise_rir) 
        infer1 = get_firstchannel_read(infer1_path)
        if choice_rir < 50:
            infer1_rir = get_firstchannel_read(infer1_rir_path)
            infer1 = add_reverb(infer1, infer1_rir)
        infer2 = get_firstchannel_read(infer2_path)
        if choice_rir < 50:
            infer2_rir = get_firstchannel_read(infer2_rir_path)
            infer2 = add_reverb(infer2, infer2_rir)
        inputs, labels = get_three_spks_noise(clean, infer1, infer2, noise, randstate)
    
    return inputs, labels

def get_auxs(clean, noise, randstate):
    snr_se = randstate.uniform(10, 30)
    gen_noise = mix_noise(clean, noise, snr_se, randstate)
    noisy = clean + gen_noise

    scale = randstate.uniform(0.3, 0.9)
    #max_amp = np.max(np.abs(noisy))
    max_amp = np.max(np.abs([noisy, clean]))
    max_amp = np.maximum(max_amp, eps)
    noisy_scale = 1. / max_amp * scale
    clean = clean * noisy_scale
    noisy = noisy * noisy_scale
    return noisy, clean

# 判断输出带噪的还是干净的aux
def generate_aux(clean, noise_path, clean_rir_path, noise_rir_path, randstate):
    choice_rir =  randstate.uniform(0, 100)
    if choice_rir < 50:
        clean_rir = get_firstchannel_read(clean_rir_path)
        clean = add_reverb(clean, clean_rir)
    
    noise = get_firstchannel_read(noise_path)
    if choice_rir < 50:
        noise_rir = get_firstchannel_read(noise_rir_path)
        noise = add_reverb(noise, noise_rir)
    
    aux_noisy, aux_clean = get_auxs(clean, noise, randstate)
    # choice clean or noisy
    choice_senario = randstate.uniform(0, 100)
    if choice_senario < 50:
        outs = aux_noisy
    else:
        outs = aux_clean
    return outs

def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({'inputs': tmp[0], 'duration': float(tmp[1])})
            else:
                path_list.append({'inputs': tmp[0]})

class AutoDataset(Dataset):

    def __init__(   self,
                    clean_scp,
                    clean_spk,
                    infer_scp,
                    noise_scp,
                    rir_scp,
                    repeat=1,
                    segment_length=4,
                    aux_segment_length=4,
                    sample_rate=16000,
                ):
        super(AutoDataset, self).__init__()

        mgr = mp.Manager()
        self.clean_list = mgr.list()
        self.infer_list = mgr.list()
        self.noise_list = mgr.list()
        self.rir_list = mgr.list()
        self.clean_spk = clean_spk # 记录id的

        self.index = mgr.list()
        self.segment_length = segment_length * sample_rate
        self.aux_segment_length = aux_segment_length * sample_rate

        pc_list = []
        p = mp.Process(target=parse_scp, args=(clean_scp, self.clean_list))
        p.start()
        pc_list.append(p)
        p = mp.Process(target=parse_scp, args=(infer_scp, self.infer_list))
        p.start()
        pc_list.append(p)
        p = mp.Process(target=parse_scp, args=(noise_scp, self.noise_list))
        p.start()
        pc_list.append(p)
        p = mp.Process(target=parse_scp, args=(rir_scp, self.rir_list))
        p.start()
        pc_list.append(p)

        for p in pc_list:
            p.join()
        # init
        self.utt2spk = dict()
        self.spk2utt = dict()
        self.init(clean_spk)
        # do chunk
        do_chunk(self.clean_list, self.index, self.segment_length, sample_rate)
        self.index *= repeat
        self.randstates = [np.random.RandomState(idx) for idx in range(3000)]
    
    def init(self, clean_spk):
        lines = open(clean_spk, 'r').readlines()
        for line in lines:
            line = line.strip().split()
            utt_path, spk_id = line[0], line[1]
            if utt_path in self.utt2spk.keys():
                raise KeyError('{} is repeated!'.format(utt_path))
            self.utt2spk[utt_path] = spk_id
            if spk_id not in self.spk2utt.keys():
                self.spk2utt[spk_id] = list()
            self.spk2utt[spk_id].append(utt_path)
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        randstate = self.randstates[(index + 11) % 3000]
        data_info, start_time = self.index[index]
        utt_path = data_info['inputs']
        spk_id = self.utt2spk[utt_path]
        spk_label = spk_id[-4:]
        spk_label = int(spk_label) - 1

        len_infer = len(self.infer_list)
        len_noise = len(self.noise_list)
        len_rir = len(self.rir_list)

        cnt_noise = randstate.randint(0, len_noise)
        cnt_clean_rir = randstate.randint(0, len_rir)
        cnt_noise_rir = randstate.randint(0, len_rir)
        cnt_infer1_rir = randstate.randint(0, len_rir)
        cnt_infer2_rir = randstate.randint(0, len_rir)

        clean_path = data_info['inputs']
        noise_path = self.noise_list[cnt_noise]['inputs']
        clean_rir_path = self.rir_list[cnt_clean_rir]['inputs']
        noise_rir_path = self.rir_list[cnt_noise_rir]['inputs']
        infer1_rir_path = self.rir_list[cnt_infer1_rir]['inputs']
        infer2_rir_path = self.rir_list[cnt_infer2_rir]['inputs']

        # choise infer1
        cnt_infer1 = randstate.randint(0, len_infer)
        infer1_path = self.infer_list[cnt_infer1]['inputs']
        while self.utt2spk[infer1_path] == spk_id:
            cnt_infer1 = randstate.randint(0, len_infer)
            infer1_path = self.infer_list[cnt_infer1]['inputs']
        
        # choise infer2
        cnt_infer2 = randstate.randint(0, len_infer)
        infer2_path = self.infer_list[cnt_infer2]['inputs']
        while self.utt2spk[infer2_path] == spk_id or self.utt2spk[infer2_path] == self.utt2spk[infer1_path]:
            cnt_infer2 = randstate.randint(0, len_infer)
            infer2_path = self.infer_list[cnt_infer2]['inputs']

        # get mix and clean
        inputs, labels = generate_data(clean_path, infer1_path, infer2_path, noise_path, clean_rir_path, infer1_rir_path,
                            infer2_rir_path, noise_rir_path, start_time, self.segment_length, self.randstates[(index + 17) % 3000])

        aux_num = 1
        new_randstate = self.randstates[(index + 29) % 3000]
        len_spk = len(self.spk2utt[spk_id])
        if len_spk == 1:
            aux_path = utt_path
            aux = get_firstchannel_read(aux_path)
        else:
            cnt_spk = new_randstate.randint(0, len_spk)
            aux_path = self.spk2utt[spk_id][cnt_spk]
            aux = get_firstchannel_read(aux_path)
            while aux_path == utt_path or aux.shape[0] <= self.aux_segment_length//3:
                cnt_spk = new_randstate.randint(0, len_spk)
                aux_path = self.spk2utt[spk_id][cnt_spk]
                aux = get_firstchannel_read(aux_path)
        
        if aux.shape[0] > self.aux_segment_length:
            st = new_randstate.randint(aux.shape[0] - self.aux_segment_length + 1)
            aux = aux[st:st+self.aux_segment_length]
        else:
            st = self.aux_segment_length - aux.shape[0]
            aux = np.concatenate([aux, aux[:st]])
            if self.aux_segment_length > aux.shape[0]:
                aux = np.pad(aux, [0, self.aux_segment_length - aux.shape[0]])
        
        aux_cnt_noise = new_randstate.randint(0, len_noise)
        aux_cnt_clean_rir = new_randstate.randint(0, len_rir)
        aux_cnt_noise_rir = new_randstate.randint(0, len_rir)
        aux_noise_path = self.noise_list[aux_cnt_noise]['inputs']
        aux_clean_rir_path = self.rir_list[aux_cnt_clean_rir]['inputs']
        aux_noise_rir_path = self.rir_list[aux_cnt_noise_rir]['inputs']

        # get aux
        aux_inputs = generate_aux(aux, aux_noise_path, aux_clean_rir_path, aux_noise_rir_path, self.randstates[(index + 47) % 3000])

        inputs = inputs.astype('float32')
        labels = labels.astype('float32')
        aux_inputs = aux_inputs.astype('float32')

        egs = {
            "mix": inputs,
            "ref": labels,
            "aux": aux_inputs,
            "spk_label": spk_label,
        }
        return egs

def worker(target_list, result_list, start, end, segment_length, sample_rate):
    for item in target_list[start:end]:
        duration = item['duration']
        length = duration * sample_rate
        if length < segment_length:
            sample_index = -1
            if length * 2 < segment_length and length * 4 > segment_length:
                sample_index = -2
            elif length * 2 > segment_length:
                sample_index = -1
            else:
                continue
            result_list.append([item, sample_index])
        else:
            sample_index = 0
            while sample_index + segment_length <= length:
                result_list.append([item, sample_index])
                sample_index += segment_length
            if sample_index < length:
                result_list.append([item, int(length - segment_length)])

def do_chunk(wav_list, index, segment_length=16000*4, sample_rate=16000, num_threads=16):
    # multiproccesing
    pc_list = []
    stride = len(wav_list) // num_threads
    if stride < 100:
        p = mp.Process(
                target=worker,
                args=(
                        wav_list,
                        index,
                        0,
                        len(wav_list),
                        segment_length,
                        sample_rate,
                )
        )
        p.start()
        pc_list.append(p)
    else:
        for idx in range(num_threads):
            if idx == num_threads - 1:
                end = len(wav_list)
            else:
                end = (idx + 1) * stride
            p = mp.Process(
                    target=worker,
                    args=(
                        wav_list,
                        index,
                        idx * stride,
                        end,
                        segment_length,
                        sample_rate,
                    )
            )
            p.start()
            pc_list.append(p)
    
    for p in pc_list:
        p.join()

def make_auto_loader(clean_scp, clean_spk, infer_scp, noise_scp, rir_scp, batch_size=8, repeat=1, 
                        num_workers=16, segment_length=4, aux_segment_length=4, sample_rate=16000):

    dataset = AutoDataset(
                clean_scp=clean_scp,
                clean_spk=clean_spk,
                infer_scp=infer_scp,
                noise_scp=noise_scp,
                rir_scp=rir_scp,
                repeat=repeat,
                segment_length=segment_length,
                aux_segment_length=aux_segment_length,
                sample_rate=sample_rate,
            )
    loader = tud.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                shuffle=True,
            )
    return loader

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

def test_loader():
    

    clean_scp = "/home/work_nfs4_ssd/xpyan/Extr/bigdata/tr/clean_tr.lst" # '/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tt/clean_small.lst'
    clean_spk = "/home/work_nfs4_ssd/xpyan/Extr/bigdata/tr/clean_tr_id.lst" # '/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tt/clean_small_id.lst'
    infer_scp = "/home/work_nfs4_ssd/xpyan/Extr/bigdata/tr/clean_tr.lst" # '/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tt/clean_small.lst'
    noise_scp = "/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tr/new_noise_tr.lst" # '/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tt/noise_small.lst'
    rir_scp = "/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tr/new_rir_tr.lst" # '/home/work_nfs4_ssd/ykjv/data/new_DNS/data_list/tt/rir_small.lst'\
    
    repeat = 1
    num_worker = 16
    segment_length = 4
    aux_segment_length = 4
    sample_rate = 16000
    batch_size = 12

    loader = make_auto_loader(
                clean_scp=clean_scp,
                clean_spk=clean_spk,
                infer_scp=infer_scp,
                noise_scp=noise_scp,
                rir_scp=rir_scp,
                batch_size=batch_size,
                repeat=repeat,
                num_workers=num_worker,
                segment_length=segment_length,
                aux_segment_length=aux_segment_length,
                sample_rate=sample_rate,
            )
    
    cnt = 0
    for egs in loader:
        # egs = load_obj(egs, th.device("cpu"))
        print(type(egs["mix"]))
        print(egs["mix"].shape)
        egs["mix"] = egs["mix"].contiguous()
        length = egs["mix"].shape[0]
        print(length)
        for i in range(length):
            wav = egs["mix"][i]
            print(type(wav))
            print(wav.shape)
            cnt = cnt + 1
            save_path = "/home/work_nfs5_ssd/mshliu/project_now/e3net/res/train_" + str(cnt) + ".wav"
            sf.write(save_path, wav.detach().numpy(), 16000)
            print(cnt)
            if cnt > 50:
                break
    print('done!')

if __name__ == "__main__":
    test_loader()
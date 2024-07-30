'''

for eval the model, pesq, stoi, si-sdr

need to install pypesq: 
https://github.com/ludlows/python-pesq

pystoi:
https://github.com/mpariente/pystoi

si-sdr:
kewang 

'''

import soundfile as sf
from pypesq import pesq
import multiprocessing as mp
import argparse
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources
import numpy as np 
import os
os.environ['OMP_NUM_THREADS'] = '2'
num_threads = 16

def audioread(path):
    wave_data, sr = sf.read(path)
    if len(wave_data.shape) >= 2:
        wave_data = wave_data[:,0]
    return wave_data, sr

def remove_dc(signal):
    """Normalized to zero mean"""
    mean = np.mean(signal)
    signal -= mean
    return signal


def pow_np_norm(signal):
    """Compute 2 Norm"""
    return np.square(np.linalg.norm(signal, ord=2))


def pow_norm(s1, s2):
    return np.sum(s1 * s2)


def si_snr(estimated, original, eps=1e-8):
    # estimated = remove_dc(estimated)
    # original = remove_dc(original)
    target = pow_norm(estimated, original) * original / pow_np_norm(original)
    noise = estimated - target
    return 10 * np.log10((pow_np_norm(target) + eps) / (pow_np_norm(noise) + eps))

def calc_sdr(ref, est):
    sdr, sir, sar, popt = bss_eval_sources(ref, est)
    return sdr[0]

number = 0
flag = False
def eval(ref_name, enh_name, nsy_name, results):
    try:
        utt_id = ref_name.split('/')[-1]
        ref, sr = audioread(ref_name)
        enh, sr = audioread(enh_name)
        nsy, sr = audioread(nsy_name)
        enh_len = enh.shape[0]
        ref_len = ref.shape[0]
        if enh_len > ref_len:
            enh = enh[:ref_len]
        else:
            ref = ref[:enh_len]
            nsy = nsy[:enh_len]
        ref_score = pesq(ref, nsy, sr)
        enh_score = pesq(ref, enh, sr)
        ref_stoi = stoi(ref, nsy, sr, extended=False)
        enh_stoi = stoi(ref, enh, sr, extended=False)
        ref_estoi = stoi(ref, nsy, sr, extended=True)
        enh_estoi = stoi(ref, enh, sr, extended=True)
        ref_sisdr = si_snr(nsy, ref)
        enh_sisdr = si_snr(enh, ref)
        ref_sdr = calc_sdr(nsy, ref)
        enh_sdr = calc_sdr(enh, ref)

        global number
        number = number + 1
        if number % 10 == 0:
            global num_threads
            print(number * num_threads)

    except Exception as e:
        print(e)
    
    results.append([utt_id, 
                    {'pesq':[ref_score, enh_score],
                     'stoi':[ref_stoi, enh_stoi],
                     'estoi':[ref_estoi, enh_estoi],
                     'si_sdr':[ref_sisdr, enh_sisdr],
                     'sdr':[ref_sdr, enh_sdr],
                    }])
                    

def main(args):
    # pathc='/home/work_nfs4_ssd/lvshubo/data/datasets/WSJ/wsj0_2mix_extr/wav8k/max/tt/s1/'###干净
    # pathe='/home/work_nfs4_ssd/ykjv/data/wsj0_2mix_extr_exp/spex+_original/'
    # pathn='/home/work_nfs4_ssd/lvshubo/data/datasets/WSJ/wsj0_2mix_extr/wav8k/max/tt/mix/'###带噪


    # 重写wav.lst，内容为每条音频的名字
    pathc='/home/work_nfs4_ssd/xpyan/Extr/minidata/tt_file/ref/'###干净
    pathe='/home/work_nfs6/zqwang/workspace/e3net/decode/e3net_wavlm_interpolation/'
    pathn='/home/work_nfs4_ssd/xpyan/Extr/minidata/tt_file/mix/'###带噪
    
    pool = mp.Pool(args.num_threads)
    mgr = mp.Manager()
    results = mgr.list()
    with open(args.result_list, 'w') as wfid:
        with open(args.wav_list) as fid:
            for line in fid:
                name = line.strip()
                pool.apply_async(
                    eval,
                    args=(
                        os.path.join(pathc,name),
                        os.path.join(pathe,name),
                        os.path.join(pathn,name),
                        results,
                    )
                    )
        pool.close()
        pool.join()
        
        ans_pesq_0 = 0
        ans_stoi_0 = 0
        ans_estoi_0 = 0
        ans_si_sdr_0 = 0
        ans_sdr_0 = 0
        ans_pesq_1 = 0
        ans_stoi_1 = 0
        ans_estoi_1 = 0
        ans_si_sdr_1 = 0
        ans_sdr_1 = 0
        ans_pesq = 0
        ans_stoi = 0
        ans_estoi = 0
        ans_si_sdr = 0
        ans_sdr = 0
        cnt = 0
        for eval_score in results:
            utt_id, score = eval_score
            pesq = score['pesq']
            stoi = score['stoi']
            estoi = score['estoi']
            si_sdr = score['si_sdr']
            sdr = score['sdr']
            wfid.writelines(
                    '{:s},{:.3f},{:.3f}, '.format(utt_id, pesq[0],pesq[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(stoi[0],stoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}, '.format(estoi[0],estoi[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}\n'.format(si_sdr[0],si_sdr[1])
                )
            wfid.writelines(
                    '{:.3f},{:.3f}\n'.format(sdr[0],sdr[1])
                )
            cnt = cnt + 1
       	    ans_pesq = ans_pesq + (pesq[1]-pesq[0])
            ans_pesq_0 = ans_pesq_0 + pesq[0]
            ans_pesq_1 = ans_pesq_1 + pesq[1]
       	    ans_stoi = ans_stoi + (stoi[1]-stoi[0])
            ans_stoi_0 = ans_stoi_0 + stoi[0]
            ans_stoi_1 = ans_stoi_1 + stoi[1]
       	    ans_estoi = ans_estoi + (estoi[1]-estoi[0])
            ans_estoi_0 = ans_estoi_0 + estoi[0]
            ans_estoi_1 = ans_estoi_1 + estoi[1]
       	    ans_si_sdr = ans_si_sdr + (si_sdr[1]-si_sdr[0])
            ans_si_sdr_0 = ans_si_sdr_0 + si_sdr[0]
            ans_si_sdr_1 = ans_si_sdr_1 + si_sdr[1]
       	    ans_sdr = ans_sdr + (sdr[1]-sdr[0])
            ans_sdr_0 = ans_sdr_0 + sdr[0]
            ans_sdr_1 = ans_sdr_1 + sdr[1]
        print("noisy pesq",ans_pesq_0/cnt)
        print("denoisy pesq",ans_pesq_1/cnt)
        print("pesq",ans_pesq/cnt)
        print("noisy stoi",ans_stoi_0/cnt)
        print("denoisy stoi",ans_stoi_1/cnt)
        print("stoi",ans_stoi/cnt)
        print("noisy estoi",ans_estoi_0/cnt)
        print("denoisy estoi",ans_estoi_1/cnt)
        print("estoi",ans_estoi/cnt)
        print("noisy si_sdr",ans_si_sdr_0/cnt)
        print("denoisy si_sdr",ans_si_sdr_1/cnt)
        print("si_sdr",ans_si_sdr/cnt)
        print("noisy sdr",ans_sdr_0/cnt)
        print("denoisy sdr",ans_sdr_1/cnt)
        print("sdr",ans_sdr/cnt)
        print("number",cnt)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav_list',
        type=str,
        default='wav.lst'
        ) 
    parser.add_argument(
        '--result_list',
        type=str,
        default='result_list'
        ) 
    parser.add_argument(
        '--num_threads',
        type=int,
        default=num_threads
        )
    args = parser.parse_args()
    main(args)

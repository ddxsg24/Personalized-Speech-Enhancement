import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
EPSILON = th.finfo(th.float32).eps


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 
# 时域loss
def sisnr_loss(inputs, labels):
    return -(si_snr(inputs, labels))
# 时域loss
def mae_loss(inputs, labels):
    l1_loss = nn.L1Loss()
    return l1_loss(inputs, labels)
    
def spk_loss(inputs, labels):
    loss = nn.CrossEntropyLoss()
    ce_loss = loss(inputs, labels)
    return ce_loss

def mse_loss(inputs, labels):
    mse = nn.MSELoss()
    loss = mse(inputs, labels)
    return loss

def kldiv_loss(inputs, labels):
    kl_loss = nn.KLDivLoss()
    inputs = F.log_softmax(inputs, dim=1)
    labels = F.softmax(labels, dim=1)
    return kl_loss(inputs, labels)


# 幅度谱loss
def get_phasen_loss(inputs, labels):
    inputs = th.stft(inputs, n_fft=512, hop_length=160,  win_length=320, window=th.hann_window(320).to(inputs.device))
    labels = th.stft(labels, n_fft=512, hop_length=160,  win_length=320, window=th.hann_window(320).to(inputs.device))

    gth_cspec = th.cat([labels[...,0].squeeze(-1), labels[...,1].squeeze(-1)], dim=1)
    est_cspec = th.cat([inputs[...,0].squeeze(-1), inputs[...,1].squeeze(-1)], dim=1)    
    gth_cspec[gth_cspec != gth_cspec] = EPSILON #将nan元素替换为EPSILON
    est_cspec[est_cspec != est_cspec] = EPSILON    

    N, F, T = gth_cspec.shape
    
    gth_mag_spec = torch.norm(gth_cspec+EPSILON, dim=1)    
    gth_pha_spec = torch.atan2(gth_cspec[:, 257:, :]+EPSILON, gth_cspec[:, :257, :]+EPSILON)                         
    est_mag_spec = torch.norm(est_cspec+EPSILON, dim=1)
    est_pha_spec = torch.atan2(est_cspec[:, 257:, :]+EPSILON, est_cspec[:, :257, :]+EPSILON)                                  

    # power compress 
    gth_cprs_mag_spec = gth_mag_spec**2
    est_cprs_mag_spec = est_mag_spec**2

    amp_loss = th.square(th.clamp(gth_cprs_mag_spec-est_cprs_mag_spec, max=0))
    
    gth_pha_spec[gth_pha_spec != gth_pha_spec] = EPSILON #将nan元素替换为EPSILON
    est_pha_spec[est_pha_spec != est_pha_spec] = EPSILON

    gth_cprs_pha_spec = gth_pha_spec**2
    est_cprs_pha_spec = est_pha_spec**2
    
    gth_cprs_pha_spec[gth_cprs_pha_spec != gth_cprs_pha_spec] = EPSILON #将nan元素替换为EPSILON
    est_cprs_pha_spec[est_cprs_pha_spec != est_cprs_pha_spec] = EPSILON

    pha_loss = th.square(th.clamp(gth_cprs_pha_spec-est_cprs_pha_spec, max=0))

    cplx_loss = th.square(th.clamp(gth_cspec-est_cspec, max=0))

    loss_1 = th.sqrt(th.mean(amp_loss*0.5))
    loss_2 = th.mean(pha_loss*0.5)
    loss_3 = th.mean(cplx_loss*0.5)

    loss = th.sqrt(loss_1 * (F // 2.0)) + th.sqrt(loss_2 * (F // 2.0)) + th.sqrt(loss_3 * (F // 2.0))
    return loss
# 幅度谱loss
def RI_Mag_Compress_Mse_Asym(inputs, labels, lamda=0.5, n_fft=1024, hop_length=480, win_length=960, window=None):
    '''
        est_real: N x 2 x F x T
    '''
    if (not window is None) and window.device != inputs.device:
        window = window.to(inputs.device)
        
    x = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    est_cspec = x.permute(0,3,1,2)
    x = torch.stft(labels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    ref_cspec = x.permute(0,3,1,2)

    est_cspec = est_cspec[:, :, 1:]
    ref_cspec = ref_cspec[:, :, 1:]

    est_mag = torch.norm(est_cspec+EPSILON, dim=1)
    est_pha = torch.atan2(est_cspec[:, 1]+EPSILON, est_cspec[:, 0]+EPSILON)
    
    ref_mag = torch.norm(ref_cspec+EPSILON, dim=1)
    ref_pha = torch.atan2(ref_cspec[:, 1]+EPSILON, ref_cspec[:, 0]+EPSILON)

    press_est_mag = torch.pow(est_mag, lamda)
    press_ref_mag = torch.pow(ref_mag, lamda)

    press_est_real = press_est_mag * torch.cos(est_pha)
    press_est_imag = press_est_mag * torch.sin(est_pha)
    press_ref_real = press_ref_mag * torch.cos(ref_pha)
    press_ref_imag = press_ref_mag * torch.sin(ref_pha)

    mag_loss = torch.square(press_est_mag - press_ref_mag)
    asym_mag_loss = torch.square(torch.clamp(press_est_mag - press_ref_mag, max=0))
    real_loss = torch.square(press_est_real - press_ref_real)
    imag_loss = torch.square(press_est_imag - press_ref_imag)
    loss = mag_loss + asym_mag_loss + real_loss + imag_loss
    
    N, F, T = loss.shape
    loss = torch.mean(loss) * F
    return loss



def kd_spk_label_loss(student, teacher, label, alpha=0.5, beta=0.5):
    """
    student: (batch x label_dim)
    teacher: (batch x label_dim)
    label: target speaker label.
    """
    if student.dim() == 1:
        student = student.unsqueeze(0)
    if teacher.dim() == 1:
        teacher = teacher.unsqueeze(0)

    B, N = student.shape

    kl_div_target = nn.KLDivLoss()
    kl_div_nontarget = nn.KLDivLoss()

    # minus 1 for using label as index 
    target_kd_loss = kl_div_target(th.log(th.abs(student[:,(label-1)])), teacher[:,(label-1)])


    stu_mask = th.ones_like(student)
    stu_mask[:,(label-1)] = 0
    tea_mask = th.ones_like(teacher)
    tea_mask[:,(label-1)] = 0

    stu_non_target = th.sum(stu_mask*student, dim=1)
    tea_non_target = th.sum(tea_mask*teacher, dim=1)
    non_target_kd_loss = kl_div_nontarget(th.log(th.abs(stu_non_target)), tea_non_target)


    kd_loss = alpha * target_kd_loss + beta * non_target_kd_loss
    return kd_loss * N


if __name__=='__main__':
    inputs = th.randn(1, 16000*4)
    ref = th.randn(1, 16000*4)

    sisnr = sisnr_loss(inputs, ref)
    phasen = get_phasen_loss(inputs, ref)
    cplx_mse_loss = RI_Mag_Compress_Mse_Asym(inputs, ref)
    
    print('sisnr', sisnr.item())
    print('phasen', phasen.item())
    print('cplx mse', cplx_mse_loss.item())   
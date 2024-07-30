import torch as th
import torch.nn as nn
import torch.nn.functional as F

from model.ecapa_zhengju import ModelWrapper as Speaker
EPSILON = th.finfo(th.float32).eps

class LSTMBlock(nn.Module):
    '''
    (fully connected + PRelU) * 2 + LayerNorm + LSTM + LayerNorm + LayerNorm
    '''

    def __init__(self, 
                 input_dim,
                 linear_dim, 
                 lstm_dim):
        super(LSTMBlock, self).__init__()

        self.fullyconnection_block = nn.Sequential(nn.Linear(input_dim, linear_dim),
                                                   nn.PReLU(),
                                                   nn.Linear(linear_dim, lstm_dim),
                                                   nn.PReLU(),
                                                   nn.LayerNorm(lstm_dim))
        
        
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, batch_first=True, bidirectional=False)
        
        self.layernorm_1 = nn.LayerNorm(lstm_dim)
        
        self.layernorm_2 = nn.LayerNorm(lstm_dim)
    
    def forward(self, input):
        x = self.fullyconnection_block(input)
        y, (h_last, c_last) = self.lstm(x)
        y = self.layernorm_1(y)
        y = y + x
        y = self.layernorm_2(y)
        return y


class E3Net(nn.Module):
    def __init__(self,
                 frame_len, 
                 frame_hop,
                 filter_num = 2048,
                 linear_dim = 1024, 
                 embedding_dim = 256,
                 lstm_dim = 256,
                 speaker_num = 1955,
                 l2_norm = True
                 ):
        super(E3Net, self).__init__()

        self.spk_model = Speaker()

        self.encoder = nn.Conv1d(in_channels = 1, out_channels = filter_num, kernel_size = frame_len, stride = frame_hop)
        self.encoder_norm = nn.Sequential(nn.PReLU(),
                                          nn.LayerNorm(filter_num))
        
        self.fullyconnection = nn.Sequential(nn.Linear(filter_num + embedding_dim, linear_dim),
                                             nn.PReLU(),
                                             nn.LayerNorm(linear_dim))

        self.lstmblock_1 = LSTMBlock(linear_dim, linear_dim, lstm_dim)
        self.lstmblock_2 = LSTMBlock(lstm_dim, linear_dim, lstm_dim)
        self.lstmblock_3 = LSTMBlock(lstm_dim, linear_dim, lstm_dim)
        self.lstmblock_4 = LSTMBlock(lstm_dim, linear_dim, lstm_dim)

        self.mask = nn.Sequential(nn.Linear(lstm_dim, filter_num), 
                                  nn.Sigmoid())

        self.decoder = nn.ConvTranspose1d(in_channels = filter_num, out_channels = 1, kernel_size = frame_len, stride = frame_hop)

        self.fc = nn.Linear(embedding_dim, speaker_num)

        self.l2_norm = l2_norm

    def forward(self, mix, aux):

        if aux.dim() ==1:
            aux = th.unsqueeze(aux, 0)
        
        e = self.spk_model(aux) # 暂时不做l2_norm
        if e.dim() == 1:
            e = th.unsqueeze(e, 0)

        if self.l2_norm:
            e = e / th.norm(e, 2, dim=1, keepdim=True)

        spk_pred = self.fc(e)
        
        if mix.dim() == 1:
            x = th.unsqueeze(mix, 0)
            x = th.unsqueeze(x, 0)
        elif mix.dim() == 2:
            x = th.unsqueeze(mix, 1)
        else:
            x = mix
        
        # N X T X 2048
        feature = self.encoder(x)
        N, _, T = feature.shape
        x = th.transpose(feature, 1, 2)
        x = self.encoder_norm(x)

        # N X D -> N X T X D 
        e = th.unsqueeze(e, 2).repeat(1, 1,  T)
        e = th.transpose(e, 1, 2)

        # N X T X (2048 + D)
        x = th.cat([x, e], -1)

        # N X T X 1024
        x = self.fullyconnection(x)

        mid_layers_feat = []

        # N X T X 256
        x = self.lstmblock_1(x)
        mid_layers_feat.append(x)
        x = self.lstmblock_2(x)
        mid_layers_feat.append(x)
        x = self.lstmblock_3(x)
        mid_layers_feat.append(x)
        x = self.lstmblock_4(x)
        mid_layers_feat.append(x)

        # N X 2048 X T
        m = self.mask(x)
        m = th.transpose(m, 1, 2)

        feature_res = feature * m

        wav = th.squeeze(self.decoder(feature_res), dim=1)
        # print(wav.shape)
        est = {"wav": wav,
               "spk_pred": spk_pred,
               "mid_layers_feat": mid_layers_feat,
               }
        return est

    def reload_spk(self, path = "/home/work_nfs4_ssd/hzhao/aslp-spknet-fork/exp/ecapa_augment_vox2/results/ecapa_augments_vox2/final.pth.tar"):
        cpt = th.load(path, map_location="cpu")
        self.spk_model.embedding_model.load_state_dict(cpt['embedding_model'], strict=False)
        # self.spk_model.embedding_model.se_res2block_list[0].dilated_conv.conv_list.insert(0, nn.Identity())
        # self.spk_model.embedding_model.se_res2block_list[1].dilated_conv.conv_list.insert(0, nn.Identity())
        # self.spk_model.embedding_model.se_res2block_list[2].dilated_conv.conv_list.insert(0, nn.Identity())          
        self.spk_model.eval()
        for param in self.spk_model.parameters():
            param.requires_grad = False
        print("successful to reload_spk: ", path)

 

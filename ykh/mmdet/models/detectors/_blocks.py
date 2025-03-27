import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

__all__ = ['BasicConv', 'Conv1x1', 'Conv3x3', 'Conv7x7', 'MaxPool2x2', 'MaxUnPool2x2', 'ConvTransposed3x3']


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d


def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU


def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)


def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)


class BasicConv(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size, pad_mode='Zero',
            bias='auto', norm=False, act=False,
            **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize() + 'Pad2d')(kernel_size // 2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias == 'auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)


class Conv7x7(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 7, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)


class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2, 2), padding=(0, 0), **kwargs)


class MaxUnPool2x2(nn.MaxUnpool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2, 2), padding=(0, 0), **kwargs)


class ConvTransposed3x3(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            bias='auto', norm=False, act=False,
            **kwargs
    ):
        super().__init__()
        seq = []
        seq.append(
            nn.ConvTranspose2d(
                in_ch, out_ch, 3,
                stride=2, padding=1,
                bias=(False if norm else True) if bias == 'auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
        # dif_as_kv
        dif = input2 - input1
        output_1 = self.cross(input1, dif)  # (Q,K,V)
        output_2 = self.cross(input2, dif)  # (Q,K,V)

        return output_1, output_2

    def cross(self, input, dif):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output


class resblock(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 2), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 2), int(outchannel / 2), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 2), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)


class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model

        # n_layers = 3
        # print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h * w, int(d_model))

        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)

        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

        self.resblock = nn.ModuleList([resblock(d_model * 2, d_model * 2) for i in range(n_layers)])

        self.LN = nn.ModuleList([nn.LayerNorm(d_model * 2) for i in range(n_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        if feature_dim == 1024 or feature_dim ==2048:
            img_feat1 = self.projection(img_feat1)
            img_feat2 = self.projection(img_feat2)
        if feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)
            img_feat2 = self.projection2(img_feat2)
        if feature_dim == 512:
            img_feat1 = self.projection3(img_feat1)
            img_feat2 = self.projection3(img_feat2)
        if feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)
            img_feat2 = self.projection4(img_feat2)

        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)
        # 1D_PE
        # position_embedding = self.embedding_1D(torch.arange(h*w, device=device).to(device)).view(h,w,self.d_model).permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)

        img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
        img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

        encoder_output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        encoder_output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

        output1 = encoder_output1
        output2 = encoder_output2
        output1_list = list()
        output2_list = list()
        for l in self.transformer:
            output1, output2 = l(output1, output2)

            output1_list.append(output1)
            output2_list.append(output2)

        # MBF
        i = 0
        output = torch.zeros((w*w, batch, self.d_model * 2)).to(device)
        for res in self.resblock:
            input = torch.cat([output1_list[i], output2_list[i]], dim=-1)
            output = output + input
            output = output.permute(1, 2, 0).view(batch, self.d_model * 2, w, w)
            output = res(output)
            output = output.view(batch, self.d_model * 2, -1).permute(2, 0, 1)
            output = self.LN[i](output)
            i = i + 1

        output=output.permute(1,2,0).view(batch,-1,w,w)
        return output


class CrossAttention(nn.Module):
    """
    Cross Transformer layer 修改版本交叉注意力机制
    """

    def __init__(self, dropout, h,w,d_model=1024, feature_dim=256,n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection_back = nn.Conv2d(d_model,feature_dim,kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))


    def forward(self, text, vis):
        #  vis_as_kv
        # dif = input2 - input1
        vis=self.projection(vis)
        batch,c,h,w=vis.size()
        device=vis.device

        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)
        # 1D_PE
        # position_embedding = self.embedding_1D(torch.arange(h*w, device=device).to(device)).view(h,w,self.d_model).permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)
        vis = vis + position_embedding  # (batch_size, d_model, h, w)

        vis = vis.view(batch, -1, h * w).permute(2, 0, 1) # (h*w, batch_size, d_model)
        text = text.permute(1, 0, 2)


        output_1 = self.cross(vis, text)  # (Q,K,V)
        output_1 = output_1.permute(1,2,0).view(batch,-1,h,w)

        output=self.projection_back(output_1)

        return output

    def cross(self, input, dif):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output
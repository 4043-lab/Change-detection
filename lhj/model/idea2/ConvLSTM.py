import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.down_stride = 2 if downsample else 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1, bias=False),
            # CondConv(out_ch // self.expansion, out_ch // self.expansion, kernel_size=3, stride=self.down_stride, padding=1, bias=False, K=4),
            nn.BatchNorm2d(out_ch // self.expansion),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(out_ch // self.expansion, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=self.down_stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.conv1_1(x)
        x = self.conv3_1(x)
        x = self.conv1_2(x)
        res = self.conv_res(res)
        out = res + x
        # out = self.relu(out)
        return out

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)

        self.conv = Bottleneck(in_ch=self.input_dim + self.hidden_dim, out_ch=self.hidden_dim, downsample=False)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        # cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        cc_i = combined_conv
        cc_f = combined_conv
        cc_o = combined_conv
        cc_g = combined_conv
        # print(cc_i.shape)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device='cuda'),
                torch.zeros(batch_size, self.hidden_dim, height, width, device='cuda'))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []  # 每一层的序列输出
        last_state_list = []  # 每一层的最后一个时间的隐藏状态h和c

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]  # 当前层的初始隐藏状态
            # print(h.shape,c.shape)
            output_inner = []

            # 执行conv_unit
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # print(output_inner[0].shape)
            layer_output = torch.stack(output_inner, dim=1)  # stack成序列
            # print(layer_output.shape)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]
        #     # print(layer_output_list[-1][:,-1,:,:,:].shape)
        #     # print(last_state_list[-1][0].shape)
        #     # print(torch.equal(layer_output_list[-1][:,-1,:,:,:],last_state_list[-1][0]))
        # return layer_output_list, last_state_list

        if not self.return_all_layers:
            return layer_output_list[-1]
        else:
            return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

if __name__ == '__main__':
    net = ConvLSTM(input_dim=64, hidden_dim=128, kernel_size=(3,3), num_layers=2, batch_first=False, bias=True, return_all_layers=False).cuda()
    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print('模型总大小为：{:.3f}MB'.format(all_size))
        # return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    getModelSize(net)
    x = torch.randn([5, 8, 64, 256, 256]).cuda()
    y = net(x)
    print(y.shape)
import torch
import torch.nn as nn
from torch.nn import functional as F

class LinearMasked(nn.Module):
    """
    Masked Linear layers used in Made.
    See Also:
        Germain et al. (2015, Feb 12) MADE:
        Masked Autoencoder for Distribution Estimation.
        Retrieved from https://arxiv.org/abs/1502.03509
    """

    def __init__(self, in_features, out_features, num_input_features, bias=True):
        """
        Parameters
        ----------
        in_features : int
        out_features : int
        num_input_features : int
            Number of features of the models input X.
            These are needed for all masked layers.
        bias : bool
        """
        super(LinearMasked, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.num_input_features = num_input_features

        # Make sure that d-values are assigned to m
        # d = 1, 2, ... D-1
        d = set(range(1, num_input_features))
        c = 0
        while True:
            c += 1
            if c > 10:
                break
            # m function of the paper. Every hidden node, gets a number between 1 and D-1
            self.m = torch.randint(1, num_input_features, size=(out_features,)).type(
                torch.int32
            )
            if len(d - set(self.m.numpy())) == 0:
                break

        self.register_buffer(
            "mask", torch.ones_like(self.linear.weight).type(torch.uint8)
        )
        
        self.cuda()
        
        
    def set_mask(self, m_previous_layer):
        """
        Sets mask matrix of the current layer.
        Parameters
        ----------
        m_previous_layer : tensor
            m values for previous layer layer.
            The first layers should be incremental except for the last value,
            as the model does not make a prediction P(x_D+1 | x_<D + 1).
            The last prediction is P(x_D| x_<D)
        """
        self.mask[...] = (m_previous_layer[:, None] <= self.m[None, :]).T

    def forward(self, x):
        if self.linear.bias is None:
            b = 0
        else:
            b = self.linear.bias
        
        return F.linear(x, self.linear.weight * self.mask, b)
    
class SequentialMasked(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

        input_set = False
        for i in range(len(args)):
            layer = self.__getitem__(i)
            if not isinstance(layer, LinearMasked):
                continue
            if not input_set:
                layer = set_mask_input_layer(layer)
                m_previous_layer = layer.m
                input_set = True
            else:
                layer.set_mask(m_previous_layer)
                m_previous_layer = layer.m

    def set_mask_last_layer(self):
        reversed_layers = filter(
            lambda l: isinstance(l, LinearMasked), reversed(self._modules.values())
        )

        # Get last masked layer
        layer = next(reversed_layers)
        prev_layer = next(reversed_layers)
        set_mask_output_layer(layer, prev_layer.m)


def set_mask_output_layer(layer, m_previous_layer):
    # Output layer has different m-values.
    # The connection is shifted one value to the right.
    layer.m = torch.arange(0, layer.num_input_features)
    layer.set_mask(m_previous_layer)
    return layer


def set_mask_input_layer(layer):
    m_input_layer = torch.arange(1, layer.num_input_features + 1)
    m_input_layer[-1] = 1e9
    layer.set_mask(m_input_layer)
    return layer

class MADE(nn.Module):
    def __init__(self, in_features, hidden_features):

        super().__init__()

        layers = [LinearMasked(in_features, hidden_features, in_features),
                  nn.ELU(),
                  LinearMasked(hidden_features, hidden_features, in_features),
                  nn.ELU(),
                  LinearMasked(hidden_features, in_features, in_features),
                  nn.Sigmoid(),] 
        layers = [i for i in layers if i is not None]
        self.layers = SequentialMasked(*layers)
        self.layers.set_mask_last_layer()

    def forward(self, x):
        return self.layers(x)

class AutoRegressiveNN(MADE):
    def __init__(self, in_features, hidden_features, context_features):
        super().__init__(in_features, hidden_features)
        self.context = nn.Linear(context_features, in_features)
        # remove MADE output layer
        del self.layers[len(self.layers) - 1]

        self.cuda()

    def forward(self, z, h):
        return self.layers(z) + self.context(h)


class IAF(nn.Module):
    """
    Inverse Autoregressive Flow
    https://arxiv.org/pdf/1606.04934.pdf
    """

    def __init__(self, size=1, context_size=1, auto_regressive_hidden=1):
        super().__init__()
        self.context_size = context_size
        self.s_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )
        self.m_t = AutoRegressiveNN(
            in_features=size,
            hidden_features=auto_regressive_hidden,
            context_features=context_size,
        )

    def determine_log_det_jac(self, sigma_t):
        return torch.log(sigma_t + 1e-6).sum(1)

    def forward(self, z, h=None):
        if h is None:
            h = torch.zeros(self.context_size)

        # Initially s_t should be large, i.e. 1 or 2.
        s_t = self.s_t(z, h) + 1.5
        sigma_t = nn.Sigmoid()(s_t)
        m_t = self.m_t(z, h)

        # log |det Jac|
        ldj = self.determine_log_det_jac(sigma_t)

        # transformation
        return sigma_t * z + (1 - sigma_t) * m_t, ldj

class Flow(nn.Module):

    def __init__(self, z_dim, c_dim, h_dim, flip, flow_blocks=4):
        super(Flow, self).__init__()
        self.flip = flip
        self.flow = nn.ModuleList([IAF(z_dim, c_dim, h_dim) for i in range(flow_blocks)])
       
    def forward(self, z, h):
        ldj = torch.zeros((z.shape[0],)).to('cuda')
        for i in self.flow:
            z, _ldj = i(z, h)
            ldj += _ldj
            if self.flip:
                z = z.flip(1)
        return z, ldj
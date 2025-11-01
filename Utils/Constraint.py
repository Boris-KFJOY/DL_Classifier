# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/7 19:36
import torch
from torch import nn

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, X):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(X)

def Spectral_Normalization(m):
    for name, layer in m.named_children():
        m.add_module(name, Spectral_Normalization(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m

def initialize_weights(m):
    try:
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
        elif isinstance(m, (nn.Linear, nn.Conv1d)):  # ✅ 新增对 Conv1d 支持
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.data.normal_(0, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()
    except Exception as e:
        # 防止 LayerNorm, GELU 等层无 weight/bias 报错
        pass

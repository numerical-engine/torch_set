import torch
import torch.nn as nn

class RowWiseLinear(nn.Linear):
    """元毎に施す線形レイヤー

    (batch_num, element_num, dim)のバッチデータを受け取り、dim毎に線形写像をする。
    """
    def forward(self, x: torch.Tensor, key_padding_mask:torch.Tensor = None)->torch.Tensor:
        """元毎に線形写像を施す
        Args:
            x (torch.Tensor): (batch_num, element_num, dim)のバッチデータ
            key_padding_mask (torch.Tensor): (batch_num, element_num)のバッチデータ

        Returns:
            torch.Tensor: (batch_num, element_num, out_features)のバッチデータ
        """
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-1)
            x = x.masked_fill(key_padding_mask, 0.)

        batch_num, element_num, dim = x.shape
        x = x.view(batch_num * element_num, dim)
        x = super().forward(x)
        return x.view(batch_num, element_num, -1)
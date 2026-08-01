import torch
import torch.nn as nn

from torch_set.snn.layer import RowWiseLinear


class MAB(nn.Module):
    """抽象クラス
    """
    def __init__(self, embed_dim:int, num_heads:int, rwlayer:nn.Module = None)->None:
        """_summary_

        Args:
            embed_dim (int): 入出力次元
            num_heads (int): ヘッド数
            rwlayer (nn.Module): 元毎に施すブロック
        """
        self.mhattn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.rwlayer = rwlayer if rwlayer is not None else RowWiseLinear(embed_dim, embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x:torch.Tensor, y:torch.Tensor, key_padding_mask:torch.Tensor = None)->torch.Tensor:
        """forward処理

        Args:
            x (torch.Tensor): shape=(batch_num, element_num, embed_dim)のバッチデータ
            y (torch.Tensor): shape=(batch_num, element_num, embed_dim)のバッチデータ
            key_padding_mask (torch.Tensor, optional): パディングマスク。shape=(batch_num, element_num)のバッチデータ。

        Returns:
            torch.Tensor: shape=(batch_num, element_num, embed_dim)のバッチデータ
        """
        h = self.mhattn(x, y, y, key_padding_mask=key_padding_mask)
        h = self.layer_norm1(h + x)
        h = self.layer_norm2(self.rwlayer(h) + h)

        return h

class SAB(MAB):
    def forward(self, x:torch.Tensor, key_padding_mask:torch.Tensor = None)->torch.Tensor:
        """forward処理

        Args:
            x (torch.Tensor): shape=(batch_num, element_num, embed_dim)のバッチデータ
            key_padding_mask (torch.Tensor, optional): パディングマスク。shape=(batch_num, element_num)のバッチデータ。

        Returns:
            torch.Tensor: shape=(batch_num, element_num, embed_dim)のバッチデータ
        """
        return super().forward(x, x, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, num_inducing_points:int, rwlayer:nn.Module = None)->None:
        """
        Args:
            embed_dim (int): 入出力次元
            num_heads (int): ヘッド数
            rwlayer (nn.Module): 元毎に施すブロック
            num_inducing_points (int): 誘導点の数
        """
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, embed_dim))
        self.mab1 = MAB(embed_dim=embed_dim, num_heads=num_heads, rwlayer=rwlayer)
        self.mab2 = MAB(embed_dim=embed_dim, num_heads=num_heads, rwlayer=rwlayer)

    def forward(self, x:torch.Tensor, key_padding_mask:torch.Tensor = None)->torch.Tensor:
        """forward処理

        Args:
            x (torch.Tensor): shape=(batch_num, element_num, embed_dim)のバッチデータ
            key_padding_mask (torch.Tensor, optional): パディングマスク。shape=(batch_num, element_num)のバッチデータ。
        Returns:
            torch.Tensor: shape=(batch_num, element_num, embed_dim)のバッチデータ
        """
        batch_num = x.shape[0]
        inducing_points = self.inducing_points.expand(batch_num, -1, -1)
        h = self.mab1(inducing_points, x, key_padding_mask=key_padding_mask)
        r = self.mab2(x, h)
        return r

class PMA(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, rwlayer_mab:nn.Module = None, rwlayer_out:nn.Module = None, num_seeds:int = 1)->None:
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.mab = MAB(embed_dim=embed_dim, num_heads=num_heads, rwlayer=rwlayer_mab)
        self.rwlayer_out = rwlayer_out if rwlayer_out is not None else RowWiseLinear(embed_dim, embed_dim)

    def forward(self, x:torch.Tensor, key_padding_mask:torch.Tensor = None)->torch.Tensor:
        """forward処理

        Args:
            x (torch.Tensor): shape=(batch_num, element_num, embed_dim)のバッチデータ
            key_padding_mask (torch.Tensor, optional): パディングマスク。shape=(batch_num, element_num)のバッチデータ。
        Returns:
            torch.Tensor: shape=(batch_num, num_seeds, embed_dim)のバッチデータ
        """
        batch_num = x.shape[0]
        seed_vectors = self.seed_vectors.expand(batch_num, -1, -1)
        x = self.rwlayer_out(x)
        r = self.mab(seed_vectors, x, key_padding_mask=key_padding_mask)
        return r
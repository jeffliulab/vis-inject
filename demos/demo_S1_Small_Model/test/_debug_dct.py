import sys, torch
sys.path.insert(0, 'd:/Projects/VisInject/demos/demo_S1_Small_Model')
from models.stego_encoder import _get_dct_matrix, dct2d_blockwise, idct2d_blockwise

N = 8
D = _get_dct_matrix(N, torch.device('cpu'), torch.float32)
err_orth = (D @ D.T - torch.eye(N)).abs().max().item()
print(f'D正交性误差: {err_orth:.2e}')

x1 = torch.rand(8)
print(f'1D往返误差: {(x1 - D.T @ (D @ x1)).abs().max():.2e}')

p = torch.rand(8, 8)
P = D @ p @ D.T
print(f'2D patch往返误差: {(p - D.T @ P @ D).abs().max():.2e}')

x = torch.rand(1, 1, 16, 16)
dct = dct2d_blockwise(x, patch_size=8)
recon = idct2d_blockwise(dct, patch_size=8)
print(f'blockwise 往返误差: {(x - recon).abs().max():.2e}')

# 检查 unfold 顺序
ps = 8
unf = x.unfold(2, ps, ps).unfold(3, ps, ps)
print(f'unfold shape: {unf.shape}')
patches = unf.contiguous().view(-1, ps, ps)
print(f'patches shape: {patches.shape}')
dct_p = D @ patches @ D.T
dct_p2 = dct_p.view(1, 1, 2, 2, 8, 8)
dct_out = dct_p2.permute(0,1,2,4,3,5).contiguous().view(1,1,16,16)
print(f'dct_out == blockwise dct: {(dct_out - dct).abs().max():.2e}')

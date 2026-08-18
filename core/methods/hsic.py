import torch

def rbf_kernel(X, Y, sigma=1.0):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    
    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return torch.exp(-dnorm2 / (2 * sigma**2))

def hsic(X, Y, sigma=1.0):
    n = X.shape[0]
    K = rbf_kernel(X, X, sigma)
    L = rbf_kernel(Y, Y, sigma)
    H = torch.eye(n, device=X.device) - torch.ones((n, n), device=X.device) / n
    return torch.trace(torch.mm(K, torch.mm(H, torch.mm(L, H)))) / (n - 1)**2

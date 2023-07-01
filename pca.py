#%%
from utils import *

#%% dual pca on the raw data
tr_neg, tr_pos = torch.load('./data/tr_val_neg_pos.pt') # data is not normalized
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized

def dual_pca(x):
    """
    x is in the shape of [features, smaples]
    """
    x = x - x.mean(1, keepdim=True)
    v, sq, vt = torch.linalg.svd(x.t()@x)
    u = x @ v @ (sq**-0.5).diag()
    return u, sq

"duel PCA for the raw data"
dn = tr_neg.reshape(tr_neg.shape[0], -1).t() # shape of [131072, 982]
dp = tr_pos.reshape(tr_pos.shape[0], -1).t()


K = 800
un, s2n = dual_pca(dn)
up, s2p = dual_pca(dp) # check how many dim to choose by s2q
proj_n = un[:, :K].t()
mean_n = dn.mean(1, keepdim=True)
proj_p = up[:, :K].t()
mean_p = dp.mean(1, keepdim=True)

"test stage, negative part"
res1 = proj_n@(test_neg.reshape(test_neg.shape[0], -1).t()- mean_n) # [K, samples]
res2 = proj_p@(test_neg.reshape(test_neg.shape[0], -1).t()- mean_p) # [K, samples]

pr1 = res1.norm(dim=0)
pr2 = res2.norm(dim=0)
n_neg = (pr1 > pr2).sum()

"postive part"
res1 = proj_n@(test_pos.reshape(test_pos.shape[0], -1).t()- mean_n) 
res2 = proj_p@(test_pos.reshape(test_pos.shape[0], -1).t()- mean_p) 

pr1 = res1.norm(dim=0)
pr2 = res2.norm(dim=0)
n_pos = (pr2 > pr1).sum()

acc = (n_pos + n_neg)/(test_neg.shape[0]*2) #0.4913 with K500
print(f'acc is {acc.item()}')

#%% PCA on the augmented data
tr_neg, tr_pos = torch.load('./data/aug_neg.pt'),\
     torch.load('./data/aug_pos.pt') # data is not normalized
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized

def tokenize(x):
    """
    x has the shape of [n, 32, 64, 64]
    """
    def my_chuck(x, chunks=2, dim=0):
        return torch.stack(torch.chunk(x,chunks,dim), dim=1)
    
    tokens = my_chuck(my_chuck(my_chuck(x,2,-1), 2,-2), 4, -3).reshape(x.shape[0],-1,8,32,32)
    return tokens

dn = tr_neg.reshape(-1, 8,32,32).reshape(982*16, -1).t()
dp = tr_pos.reshape(-1, 8,32,32).reshape(982*16, -1).t() #[feature, sample]

def pca(x):
    """
    [feature, sample]
    """
    x = x - x.mean(1, keepdim=True)
    u, sq, vt = torch.linalg.svd(x@x.t())
    return  u, sq

K = 500
un, s2n = pca(dn)
up, s2p = pca(dp) # check how many dim to choose by s2q
proj_n = un[:, :K].t()[None, None]
mean_n = dn.mean(1, keepdim=True)
proj_p = up[:, :K].t()[None, None]
mean_p = dp.mean(1, keepdim=True)

test_neg_tokens, test_pos_tokens = tokenize(test_neg), tokenize(test_pos) #[n, 16, 8, 32, 32]
tnt = test_neg_tokens.reshape(174, 16, -1, 1)
tpt = test_pos_tokens.reshape(174, 16, -1, 1)

res1 = (proj_n @ (tnt - mean_n)).squeeze().norm(dim=-1) #[n, 16, K]
res2 = (proj_p @ (tnt - mean_p)).squeeze().norm(dim=-1)

"""theoretically one positive in the token is postive
but then the false alarm rate will be too high.
So a threshold is needed
"""
thre = 5
res = res1>res2
acc_n = (res.sum(1) <= thre).sum()

res1 = (proj_n @ (tpt - mean_n)).squeeze().norm(dim=-1) #[n, 16, K]
res2 = (proj_p @ (tpt - mean_p)).squeeze().norm(dim=-1)
res = res2>res1
acc_p = (res.sum(1) >= 11).sum()

acc = (acc_n + acc_p)/(test_neg.shape[0]*2) 
print(f'acc is {acc.item()}') #0.9224


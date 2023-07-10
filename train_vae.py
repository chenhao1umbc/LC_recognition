#%%
from utils import *
# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from models import VAE, MLP

#%% train with aug data
tr_pos = torch.load('./data/aug_pos.pt') # data is not normalized
tr_neg = torch.load('./data/aug_neg.pt')
tr_neg = tr_neg/(tr_neg.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
tr_pos = tr_pos/(tr_pos.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)

n_tr = int(tr_neg.shape[0]*0.85)
d_tr = torch.cat((tr_neg[:n_tr].reshape(-1, 8, 32, 32) \
                  , tr_pos[:n_tr].reshape(-1, 8, 32, 32)))
l_tr = torch.cat((torch.zeros(n_tr*16), torch.ones(n_tr*16)))
data = Data.TensorDataset(d_tr, l_tr)
tr = Data.DataLoader(data, batch_size=64, shuffle=True)

n_val = tr_neg.shape[0]- n_tr
d_val = torch.cat((tr_neg[n_tr:].reshape(-1, 8, 32, 32),\
                    tr_pos[n_tr:].reshape(-1, 8, 32, 32)))
l_val = torch.cat((torch.zeros(n_val*16), torch.ones(n_val*16)))
data = Data.TensorDataset(d_val, l_val)
val = Data.DataLoader(data, batch_size=64, shuffle=False)

vae = VAE().cuda()
temp = torch.load('./res/vae_unsupervised/model_epoch200.pt')
vae.load_state_dict(temp.module.state_dict())
mlp = MLP().cuda()
opt1 = torch.optim.RAdam(vae.parameters(), lr=1e-4)
opt2 = torch.optim.RAdam(mlp.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

tr_loss, val_loss, acc_all = [], [], []
for epoch in range(201):
    vae.train()
    mlp.train()
    temp = []
    for i, (x,y) in enumerate(tr):
        opt1.zero_grad()
        opt2.zero_grad()
        x_cuda, y_cuda = x.cuda(), y.cuda()
        x_hat, mu, logvar, sources = vae(x_cuda)
        latent = torch.cat((mu, logvar), dim = 1)
        y_hat = mlp(latent)
        loss = loss_func(y_hat, y_cuda.to(torch.long)) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=10)
        opt1.step()
        opt2.step()
        torch.cuda.empty_cache()
            
        temp.append(loss.cpu().item()/x.shape[0])
    tr_loss.append(sum(temp)/len(temp))

    #validation
    vae.eval()
    mlp.eval()
    with torch.no_grad():
        temp, acc = [], []
        for i, (x,y) in enumerate(val):
            x_cuda, y_cuda = x.cuda(), y.cuda()
            x_hat, mu, logvar, sources = vae(x_cuda)
            latent = torch.cat((mu, logvar), dim = 1)
            y_hat = mlp(latent)
            loss = loss_func(y_hat, y_cuda.to(torch.long)) 
            temp.append(loss.cpu().item()/x.shape[0])
            counter = ((y_hat.argmax(dim=-1)- y_cuda) == 0).sum()
            acc.append(counter)

        val_loss.append(sum(temp)/len(temp))
        acc_all.append(sum(acc)/d_val.shape[0])
    print(f'acc at epoch {epoch}:', acc_all[-1])
    if acc_all[-1] == max(acc_all):
        torch.save((vae, mlp), './res/vae/best_vae_aug.pt')

    if epoch%20 == 0 and epoch > 50:
        print(epoch)
        plt.figure()
        plt.plot(tr_loss, '-o')
        plt.plot(val_loss, '--^')
        plt.legend(['Training', 'Validation'])
        plt.title(f'Loss fuction at epoch {epoch}')
        plt.savefig('./res/vae/LossFun_aug.png')

        plt.figure()
        plt.plot(tr_loss[-50:], '-o')
        plt.plot(val_loss[-50:], '--^')
        plt.legend(['Training', 'Validation'])
        plt.title(f'Last 50 loss function values at {epoch}')
        plt.savefig(f'./res/vae/Last_50 at {epoch}_aug.png')
        plt.close('all')

        torch.save([tr_loss, val_loss], './res/vae/tr_val_loss_aug.pt')
        torch.save((vae, mlp), f'./res/vae/model_epoch{epoch}_aug.pt')
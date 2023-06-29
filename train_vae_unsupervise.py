#%%
from utils import *
from models import VAE, Loss

"both of the aug_neg/pos are the mixture data"
aug_neg = torch.load('data/aug_neg.pt') # shape of aug_neg is [n,16,8,32,32]
aug_pos = torch.load('data/aug_pos.pt')  # shape of aug_pos is [n,16,8,32,32]

d = torch.cat((aug_neg, aug_pos)).reshape(-1, 8, 32, 32)
ntr, nval = int(d.shape[0]*0.8), int(d.shape[0]*0.1)
data = Data.TensorDataset(d[:ntr])
tr = Data.DataLoader(data, batch_size=96, shuffle=True)
data = Data.TensorDataset(d[ntr:ntr+nval])
val = Data.DataLoader(data, batch_size=96)
data = Data.TensorDataset(d[ntr+nval:])
te = Data.DataLoader(data, batch_size=96)

model = VAE().cuda()
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
loss_func = Loss(sources=2,likelihood='gauss')

#%%
tr_loss, val_loss = [], []
for epoch in range(200):
    model.train()
    temp = []
    for i, (x,) in enumerate(tr):
        optimizer.zero_grad()
        x_cuda = x.cuda()
        x_hat, mu, logvar, sources = model(x_cuda)
        loss, Recon, KLD = loss_func(x_cuda, x_hat, mu, logvar) #loss = Rec + beta*KLD
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        temp.append(loss.cpu().item()/x.shape[0])
    tr_loss.append(sum(temp)/len(temp))
    
    #validation
    model.eval()
    with torch.no_grad():
        temp = []
        for i, (x,) in enumerate(val):
            x_cuda = x.cuda()
            x_hat, mu, logvar, sources = model(x_cuda)
            loss, Recon, KLD = loss_func(x_cuda, x_hat, mu, logvar)
            temp.append(loss.cpu().item()/x.shape[0])
        val_loss.append(sum(temp)/len(temp))

    if epoch%5 == 0:
        print(epoch)
        plt.figure()
        plt.plot(tr_loss, '-o')
        plt.plot(val_loss, '--')
        plt.legend(['Training', 'Validation'])
        plt.title(f'Loss fuction at epoch {epoch}')
        plt.savefig('./res/LossFun.png')

        torch.save(model, './res/model_epoch{epoch}.pt')
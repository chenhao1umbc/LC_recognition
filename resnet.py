#%%
from utils import *

#%% using the raw data with 2d conv, without pretrained
"process data"
tr_neg, tr_pos = torch.load('./data/tr_val_neg_pos.pt') # data is not normalized
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized
n_tr = int(tr_neg.shape[0]*0.85)
d_tr = torch.cat((tr_neg[:n_tr], tr_pos[:n_tr]))
l_tr = torch.cat((torch.zeros(n_tr), torch.ones(n_tr)))
data = Data.TensorDataset(d_tr, l_tr)
tr = Data.DataLoader(data, batch_size=32, shuffle=True)

n_val = tr_neg.shape[0]- n_tr
d_val = torch.cat((tr_neg[n_tr:], tr_pos[n_tr:]))
l_val = torch.cat((torch.zeros(n_val), torch.ones(n_val)))
data = Data.TensorDataset(d_val, l_val)
val = Data.DataLoader(data, batch_size=32, shuffle=False)

"Train the model"
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = model.cuda()

optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
loss_func = nn.BCELoss()

tr_loss, val_loss = [], []
for epoch in range(201):
    model.train()
    temp = []
    for i, (x, y) in enumerate(tr):
        optimizer.zero_grad()
        x_cuda, y_cuda = x.cuda(), y.cuda()
        y_hat = model(x_cuda)
        loss = loss_func(y_cuda, y_hat) #loss = Rec + beta*KLD
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
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
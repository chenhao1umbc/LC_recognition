#%%
from utils import *

#%% using the raw data with 2d conv, without pretrained
"process data"
tr_neg, tr_pos = torch.load('./data/tr_val_neg_pos.pt') # data is not normalized
tr_neg = tr_neg/(tr_neg.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)
tr_pos = tr_pos/(tr_pos.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)

n_tr = int(tr_neg.shape[0]*0.85)
d_tr = torch.cat((tr_neg[:n_tr], tr_pos[:n_tr]))
l_tr = torch.cat((torch.zeros(n_tr), torch.ones(n_tr)))
data = Data.TensorDataset(d_tr, l_tr)
tr = Data.DataLoader(data, batch_size=64, shuffle=True)

n_val = tr_neg.shape[0]- n_tr
d_val = torch.cat((tr_neg[n_tr:], tr_pos[n_tr:]))
l_val = torch.cat((torch.zeros(n_val), torch.ones(n_val)))
data = Data.TensorDataset(d_val, l_val)
val = Data.DataLoader(data, batch_size=64, shuffle=False)

"Train the model"
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = model.cuda()

#%%
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
loss_func = nn.CrossEntropyLoss()

tr_loss, val_loss = [], []

for epoch in range(1):
    model.train()
    temp = []
    for i, (x, y) in enumerate(tr):
        optimizer.zero_grad()
        x_cuda, y_cuda = x.cuda(), y.cuda()
        y_hat = model(x_cuda).squeeze()
        loss = loss_func(y_hat, y_cuda.to(torch.long)) #loss = Rec + beta*KLD
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
        for i, (x, y) in enumerate(val):
            x_cuda, y_cuda = x.cuda(), y.cuda()
            y_hat = model(x_cuda).squeeze()
            loss = loss_func(y_hat, y_cuda.to(torch.long))
            temp.append(loss.cpu().item()/x.shape[0])
        val_loss.append(sum(temp)/len(temp))


torch.save([tr_loss, val_loss], './res/resnet_tr_val_loss.pt')
plt.figure()
plt.plot(tr_loss, '-o')
plt.plot(val_loss, '--^')
plt.legend(['Training, Validation'])
plt.savefig('./res/resnet_loss_func_values.png')


# TODO calc accuracy and save the trained model BCE? test? aug?


#%%
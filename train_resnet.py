#%%
from utils import *
# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#%% using the aug data with 2d conv, without pretrained
"process data"
test_neg, test_pos = torch.load('./data/tr_val_neg_pos.pt') # data is not normalized
def tokenize(x):
    """
    x has the shape of [n, 32, 64, 64]
    """
    def my_chuck(x, chunks=2, dim=0):
        return torch.stack(torch.chunk(x,chunks,dim), dim=1)
    
    tokens = my_chuck(my_chuck(my_chuck(x,2,-1), 2,-2), 4, -3).reshape(x.shape[0],-1,8,32,32)
    return tokens

test_neg_tokens, test_pos_tokens = tokenize(test_neg), tokenize(test_pos) #[n, 16, 8, 32, 32]
core_neg = test_neg[:,12:20, 16:48, 16:48]
core_pos = test_pos[:,12:20, 16:48, 16:48]
test_neg_tokens = test_neg_tokens + core_neg[:,None]
test_pos_tokens = test_pos_tokens + core_pos[:,None]
tr_neg = test_neg_tokens/(test_neg_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
tr_pos = test_pos_tokens/(test_pos_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
# tr_pos = torch.load('./data/aug_pos.pt') # data is not normalized
# tr_neg = torch.load('./data/aug_neg.pt')
# tr_neg = tr_neg/(tr_neg.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
# tr_pos = tr_pos/(tr_pos.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)

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
model.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = model.cuda()

optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
loss_func = nn.CrossEntropyLoss()

tr_loss, val_loss, acc_all = [], [], []
for epoch in range(201):
    model.train()
    temp = []
    for i, (x, y) in enumerate(tr):
        optimizer.zero_grad()
        x_cuda, y_cuda = x.reshape(-1, 8, 32, 32).cuda(), y.cuda()
        y_hat = model(x_cuda).squeeze().reshape(-1,16,1000).sum(dim=1)
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
        temp, acc = [], []
        for i, (x, y) in enumerate(val):
            x_cuda, y_cuda = x.reshape(-1, 8, 32, 32).cuda(), y.cuda()
            y_hat = model(x_cuda).squeeze().reshape(-1,16,1000).sum(dim=1)
            loss = loss_func(y_hat, y_cuda.to(torch.long))
            temp.append(loss.cpu().item()/x.shape[0])
            counter = ((y_hat.argmax(dim=-1)- y_cuda) == 0).sum()
            acc.append(counter)

        val_loss.append(sum(temp)/len(temp))
        acc_all.append(sum(acc)/d_val.shape[0])
        print(f'acc at epoch {epoch}:', acc_all[-1])
        if acc_all[-1] == max(acc_all):
            torch.save(model, './res/resnet/best_resnet_aug2_new.pt')

    if epoch > 50 and epoch%20 == 0:
        plt.plot(tr_loss[-50:], '-o')
        plt.plot(val_loss[-50:], '--^')
        plt.legend(['Training', 'Validation'])
        plt.savefig(f'./res/resnet/resnet_loss_{epoch}_aug2_new.png')
        plt.close('all')

torch.save([tr_loss, val_loss], './res/resnet/resnet_tr_val_loss_aug2_new.pt')
torch.save(acc_all, './res/resnet/val_acc_aug2_new.pt')
plt.figure()
plt.plot(tr_loss, '-o')
plt.plot(val_loss, '--^')
plt.legend(['Training', 'Validation'])
plt.savefig('./res/resnet/resnet_loss_func_values_aug2_new.png')
plt.close('all')
print('done')
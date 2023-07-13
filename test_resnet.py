#%%
from utils import *
# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#%% test on raw data
"process data"
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized
test_neg = test_neg/(test_neg.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)
test_pos = test_pos/(test_pos.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)

n_test = test_neg.shape[0]
d_test = torch.cat((test_neg, test_pos))
l_test = torch.cat((torch.zeros(n_test), torch.ones(n_test)))
data = Data.TensorDataset(d_test, l_test)
te = Data.DataLoader(data, batch_size=64, shuffle=False)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model = torch.load('./res/resnet/best_resnet_pre.pt')

model.eval()
with torch.no_grad():
    temp, acc = [], []
    for i, (x, y) in enumerate(te):
        x_cuda, y_cuda = x.cuda(), y.cuda()
        y_hat = model(x_cuda).squeeze()
        counter = ((y_hat.argmax(dim=-1)- y_cuda) == 0).sum()
        acc.append(counter)
acc_test = sum(acc)/d_test.shape[0]
print('test acc is ', acc_test) # 0.7241


#%% test on augmented data using threshold
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized
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
test_neg = test_neg_tokens/(test_neg_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
test_pos = test_pos_tokens/(test_pos_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)

n_test = test_neg.shape[0]
d_test = torch.cat((test_neg, test_pos))
l_test = torch.cat((torch.zeros(n_test), torch.ones(n_test)))
data = Data.TensorDataset(d_test, l_test)
te = Data.DataLoader(data, batch_size=64, shuffle=False)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = torch.load('./res/resnet/best_resnet_aug.pt')

threshold = 8
model.eval()
with torch.no_grad():
    temp, acc = [], []
    for i, (x, y) in enumerate(te):
        x_cuda, y_cuda = x.reshape(-1,8,32,32).cuda(), y.cuda()
        y_hat16 = model(x_cuda).squeeze()
        y_hat = y_hat16.reshape(-1,16, 1000)
        l_hat = y_hat.argmax(dim=-1).sum(dim=-1)
        l_hat[l_hat<=threshold] = 0
        l_hat[l_hat>threshold] = 1
        counter = ((l_hat- y_cuda) == 0).sum()
        acc.append(counter)
        print(y_hat.argmax(dim=-1))

acc_test = sum(acc)/d_test.shape[0]
print('test acc is ', acc_test) 

#%% aug data without not using threshold
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized
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
test_neg = test_neg_tokens/(test_neg_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)
test_pos = test_pos_tokens/(test_pos_tokens.abs().amax(dim=(1,2,3,4), keepdim=True) + 1e-5)

n_test = test_neg.shape[0]
d_test = torch.cat((test_neg, test_pos))
l_test = torch.cat((torch.zeros(n_test), torch.ones(n_test)))
data = Data.TensorDataset(d_test, l_test)
te = Data.DataLoader(data, batch_size=64, shuffle=False)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.conv1 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
model = torch.load('./res/resnet/best_resnet_aug2_new.pt')

model.eval()
with torch.no_grad():
    temp, acc = [], []
    for i, (x, y) in enumerate(te):
        x_cuda, y_cuda = x.reshape(-1,8,32,32).cuda(), y.cuda()
        y_hat = model(x_cuda).squeeze().reshape(-1,16,1000).sum(dim=1)
        counter = ((y_hat.argmax(dim=-1)- y_cuda) == 0).sum()
        acc.append(counter)

acc_test = sum(acc)/d_test.shape[0]
print('test acc is ', acc_test) #0.7586
#%%
from utils import *
from models import Model3d
# torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

"process data"
test_neg, test_pos = torch.load('./data/test_neg_pos.pt') # data is not normalized
test_neg = test_neg/(test_neg.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)
test_pos = test_pos/(test_pos.abs().amax(dim=(1,2,3), keepdim=True) + 1e-5)

n_test = test_neg.shape[0]
d_test = torch.cat((test_neg, test_pos))
l_test = torch.cat((torch.zeros(n_test), torch.ones(n_test)))
data = Data.TensorDataset(d_test, l_test)
te = Data.DataLoader(data, batch_size=64, shuffle=False)

model = torch.load('./res/3d/best_3d.pt')

model.eval()
with torch.no_grad():
    temp, acc = [], []
    for i, (x, y) in enumerate(te):
        x_cuda, y_cuda = x[:,None].cuda(), y.cuda()
        y_hat = model(x_cuda).squeeze()
        y_hat[y_hat>0.5] = 1
        y_hat[y_hat<=0.5] = 0
        counter = ((y_hat - y_cuda) == 0).sum()
        acc.append(counter)
acc_test = sum(acc)/d_test.shape[0]
print('test acc is ', acc_test) # 0.7328
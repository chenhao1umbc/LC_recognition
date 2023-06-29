#%%
from utils import *

tr_loss = torch.arange(10)
val_loss = torch.arange(10)-3

plt.figure()
plt.plot(tr_loss, '-o')
plt.plot(val_loss, '--^')
plt.legend(['Training', 'Validation'])

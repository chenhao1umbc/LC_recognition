#%%
from utils import *
from models import VAE, Loss
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
torch.autograd.set_detect_anomaly(True)

d = torch.load('./data/unsup_val.pt')
data = Data.TensorDataset(d)
val = Data.DataLoader(data, batch_size=96)

model = torch.load('./res/model_epoch190.pt')
loss_func = Loss(sources=2,likelihood='gauss')

#%%

val_loss = []
model.eval()
with torch.no_grad():
    temp = []
    for i, (x,) in enumerate(val):
        x_cuda = x.cuda()
        x_hat, mu, logvar, sources = model(x_cuda)
        loss, Recon, KLD = loss_func(x_cuda, x_hat, mu, logvar)
        temp.append(loss.cpu().item()/x.shape[0])
        break
    val_loss.append(sum(temp)/len(temp))

#%% check the reconstruction
n = 10

fig, axs = plt.subplots(4, 2, figsize=(12, 6))
axs = axs.ravel()
images = []
for i, image in enumerate(x_hat[n].cpu()):
    axs[i].axis('off')
    img = axs[i].imshow(image)
    images.append(img)  # Store the plotted image
    axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)
cbar_axs = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axs]
cbar = [fig.colorbar(img, cax=cax) for img, cax in zip(images, cbar_axs)]
plt.tight_layout()  # Adjust the layout to prevent overlapping
fig.text(.5, .0001, "x estimation", ha='center')
plt.show()


fig, axs = plt.subplots(4, 2, figsize=(12, 6))
axs = axs.ravel()
images = []
for i, image in enumerate(x[n].cpu()):
    axs[i].axis('off')
    img = axs[i].imshow(image)
    images.append(img)  # Store the plotted image
    axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)
cbar_axs = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axs]
cbar = [fig.colorbar(img, cax=cax) for img, cax in zip(images, cbar_axs)]
plt.tight_layout()  # Adjust the layout to prevent overlapping
fig.text(.5, .0001, "Ground truth", ha='center')
plt.show()


#%% check the sources
n = 10

fig, axs = plt.subplots(4, 2, figsize=(12, 6))
axs = axs.ravel()
images = []
for i, image in enumerate(sources[n, 0].cpu()):
    axs[i].axis('off')
    img = axs[i].imshow(image)
    images.append(img)  # Store the plotted image
    axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)
cbar_axs = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axs]
cbar = [fig.colorbar(img, cax=cax) for img, cax in zip(images, cbar_axs)]
plt.tight_layout()  # Adjust the layout to prevent overlapping
fig.text(.5, .0001, "Source 1", ha='center')
plt.show()


fig, axs = plt.subplots(4, 2, figsize=(12, 6))
axs = axs.ravel()
images = []
for i, image in enumerate(sources[n, 1].cpu()):
    axs[i].axis('off')
    img = axs[i].imshow(image)
    images.append(img)  # Store the plotted image
    axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)
cbar_axs = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axs]
cbar = [fig.colorbar(img, cax=cax) for img, cax in zip(images, cbar_axs)]
plt.tight_layout()  # Adjust the layout to prevent overlapping
fig.text(.5, .0001, "Source 1", ha='center')
plt.show()

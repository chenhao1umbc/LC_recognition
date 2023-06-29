#%%
from utils import *

#%%
neg_all, pos_all = torch.load('./data/neg_pos.pt')
def my_chuck(x, chunks=2, dim=0):
    return torch.stack(torch.chunk(x,chunks,dim), dim=1)

def show_data(data, n):
    "From D"
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    axs = axs.ravel()

    for i, image in enumerate(data[n]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

    "From H"
    fig, axs = plt.subplots(8,8, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(data[n].permute(1,0,2)):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

    "From W"
    fig, axs = plt.subplots(8,8, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(data[n].permute(2,0,1)):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

view_data = False
if view_data:
    show_data(neg_all, 0)
    show_data(pos_all, 200)

torch.manual_seed(1)
n_tr_val = int(neg_all.shape[0]*0.85)
ind = torch.randperm(neg_all.shape[0])
neg_shuffle, pos_shuffle = neg_all[ind], pos_all[ind]
pos, neg = neg_shuffle[:n_tr_val], pos_shuffle[:n_tr_val]
torch.save( [neg_shuffle[n_tr_val:], pos_shuffle[n_tr_val:]], './data/test_neg_pos.pt' )
torch.save( [neg_shuffle[:n_tr_val], pos_shuffle[:n_tr_val]], './data/tr_val_neg_pos.pt' )

#%% Get the augmented positive
core = pos[:,12:20, 16:48, 16:48] # must contains tumor
tokens = my_chuck(my_chuck(my_chuck(pos,2,-1), 2,-2), 4, -3).reshape(pos.shape[0],-1,8,32,32)
aug_pos = tokens + core[:,None]

view_data = False
if view_data:
    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(tokens[20,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(aug_pos[100,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
torch.save(aug_pos, './data/aug_pos.pt')

#%%  Get the augmented negative (still as positive)
tokens = my_chuck(my_chuck(my_chuck(neg,2,-1), 2,-2), 4, -3).reshape(neg.shape[0],-1,8,32,32)
aug_neg_core = tokens + core[:,None]

view_data = False
if view_data:
    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(tokens[20,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()


    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(aug_neg[20,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
torch.save(aug_neg_core, './data/aug_neg+core.pt')

#%%  Get the augmented negative (still as positive)
aug_neg = my_chuck(my_chuck(my_chuck(neg,2,-1), 2,-2), 4, -3).reshape(neg.shape[0],-1,8,32,32)

view_data = False
if view_data:
    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(tokens[20,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()


    fig, axs = plt.subplots(4, 2, figsize=(12, 6))
    axs = axs.ravel()
    for i, image in enumerate(aug_neg[20,0]):
    # for i, image in enumerate(tokens[100,0]):
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(f'Slice {i+1}')  # Set a title for each image (optional)

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
torch.save(aug_neg, './data/aug_neg.pt')

#%%
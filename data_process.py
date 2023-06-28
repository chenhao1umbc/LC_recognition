#%%
from utils import *

#%%
neg, pos = torch.load('./data/neg_pos.pt')
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

show_data(neg, 0)
show_data(pos, 200)

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
# torch.save(aug_pos, 'aug_pos.pt')

#%%  Get the augmented negative (still as positive)
tokens = my_chuck(my_chuck(my_chuck(neg,2,-1), 2,-2), 4, -3).reshape(neg.shape[0],-1,8,32,32)
aug_neg = tokens + core[:,None]

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
torch.save(aug_neg, 'aug_neg.pt')
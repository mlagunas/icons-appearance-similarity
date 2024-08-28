import model_icons
import torch
import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# set the device to either GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pairwise_dist(x):
    """ Computes pairwise distances between features
        x : torch tensor with shape Batch x n_features
    """
    n = x.size(0)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, x, x.t())
    dist = dist.clamp(min=1e-12).sqrt()  # numerical stability
    return dist


def get_features(model, loader):
    """ iterates the images in the 'loader' and obtaines the feature vectors given by
        the 'model'. At the end it returns all the feature vectors colledcted together
        with the input images
    """
    # iterate over all the dataset
    with torch.no_grad():  # avoid memory usage to store gradients
        all_icons_f = []  # store here the features
        all_icons = []  # store here the input images
        for icons, _ in tqdm(loader):
            # move icons to the correct device
            icons = icons.to(device)
            plot_tensor(icons[0], 'test', 'test_folder')

            # get icon features
            icons_f = model(icons)

            # store the features and images
            for icon_f, icon in zip(icons_f, icons):
                all_icons_f.append(icon_f)
                # WARN: with a large number of images you might run out of RAM
                all_icons.append(icon)

        # move stored features and icons from list to torch tensor
        all_icons_f = torch.stack(all_icons_f)
        all_icons = torch.stack(all_icons)

    return all_icons_f, all_icons


def plot_tensor(tensor, title="", folder=None):
    """ plots a torch tensor as an image. 'title' will be the image title
        and 'folder' the folder where it will be stored
    """
    # move to numpy and change shape from CxWxH to WxHxC
    np_array = tensor.cpu().numpy().transpose((1, 2, 0))

    plt.imshow(np_array.squeeze(), cmap='Greys_r')
    plt.title(title)

    # if a folder is given store in the folder otherwise plot
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, title + '.png')
        plt.savefig(path)
    else:
        plt.show()


def plot_k_closer(reference_ix, k, dist, images, folder=None):
    # plot reference image
    reference = images[reference_ix]
    plot_tensor(reference, 'reference', folder)

    # obtain closer images to the reference according to the features
    min_dist, min_idx = torch.topk(dist[reference_ix], k=k, largest=False)

    # plot closer images to the reference
    for i in range(k):
        dist, idx = min_dist[i], min_idx[i]
        icon = images[idx]
        plot_tensor(icon, '%d-dist:%2.2f' % (i + 1, dist), folder)


if __name__ == '__main__':
    # load dataset
    dataset_path = 'small_dataset'
    trf = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('L')),  # move PIL icon to B/W
        transforms.Resize(size=(180, 180)),  # resize icon to be 180x180
        transforms.ToTensor(),  # move read image to torch tensor
    ])
    loader = DataLoader(ImageFolder(dataset_path, transform=trf),
                        batch_size=16, num_workers=4,
                        drop_last=False, shuffle=False)

    # load model, move to correct device and set in evaluation
    model = model_icons.model
    model.load_state_dict(torch.load('model_icons.pth'))
    model = model.to(device)
    model.eval()

    # get all features and icons
    all_icons_f, all_icons = get_features(model, loader)

    # get pairwise distances between features
    all_icons_dist = pairwise_dist(all_icons_f)
    # make diagonal a big number
    all_icons_dist[range(len(all_icons_dist)), range(len(all_icons_dist))] = 9999

    # plot all the images
    reference_ix = 5
    n_close_elems = 5
    out_folder = 'similar_icons'
    plot_k_closer(reference_ix, k=n_close_elems,
                  dist=all_icons_dist, images=all_icons, folder=out_folder)

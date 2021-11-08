from fastai.vision.all import *
from fastbook import *
from sklearn.model_selection import train_test_split
matplotlib.rc('image', cmap='Greys')
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

def give_steady_mm1(lam, mu, max_stead = 100):
    rho = lam/mu
    power_arr = torch.arange(max_stead)
    return ((1-rho)*rho**power_arr)

def sigmoid(x): return 1/(1+torch.exp(-x))

def queue_loss(predictions, targes):
    predictions = m(predictions)
    return ((predictions-targes)**2).mean()

def feed_forword(xb, model):
    return m(model(xb))
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()

def KL(preds, trgs):
    m = nn.Softmax(dim=1)
    preds = m(preds)
    return (preds*torch.log(preds/trgs)).mean()

def main():
    path_train = '../data/deep_queue_mm1_train.pkl'
    path_valid = '../data/deep_queue_mm1_valid.pkl'

    if os.path.exists(path_train):
        train_x, train_y = pkl.load(open(path_train, 'rb'))
        test_x, test_y = pkl.load(open(path_valid, 'rb'))

    else:
        max_stead = 50
        max_vals = 400
        eps = 0.1
        y_data = torch.empty((0, max_vals))
        lam_vals = torch.linspace(0.0001, 0.99999, max_vals)
        x_data = torch.empty((0, 2))
        for lam_ind, lam_val in tqdm(enumerate(lam_vals)):
            mu_vals = torch.linspace(lam_val + eps, 1, max_vals - lam_ind)
            for mu in mu_vals:
                curr_stead = give_steady_mm1(lam_val, mu, max_stead).reshape(1, max_stead)
                if curr_stead.sum() > 0.999:
                    #         curr_arr = np.append([lam_val,mu], curr_stead).reshape(1,curr_stead.shape[0]+2)

                    if y_data.shape[0] == 0:
                        y_data = curr_stead
                        x_data = torch.tensor([lam_val, mu]).reshape(1, 2)

                    else:
                        y_data = torch.cat((y_data, curr_stead), axis=0)
                        x_data = torch.cat((x_data, torch.tensor([lam_val, mu]).reshape(1, 2)), axis=0)

        train_x, x_test, train_y, y_test = train_test_split(x_data, y_data)

        pkl.dump((train_x, train_y), open(path_train, 'wb'))
        pkl.dump((test_x, test_y), open(path_valid, 'wb'))

    dset = list(zip(train_x, train_y))
    valid_dset = list(zip(x_test, y_test))
    x, y = dset[0]

    dl = DataLoader(dset, batch_size=256)
    valid_dl = DataLoader(valid_dset, batch_size=256)
    m = nn.Softmax(dim=1)

    simple_net = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 30),
        nn.ReLU(),
        nn.Linear(30, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50)
    )

    dls = DataLoaders(dl, valid_dl)

    learn = Learner(dls, simple_net, opt_func=SGD,
                    loss_func=queue_loss, metrics=queue_loss)

    learn.dls.to('cuda')

    learn.fit(150, lr=0.1)



    torch.save(learn.model.state_dict(), './Kash')

    xb, yb = first(valid_dl)

    a = learn.get_preds(dl=[(xb, yb)])

    import matplotlib.pyplot as plt
    exmaple = torch.randint(0, xb.shape[0], (1,))[0].item()
    max_number = 10
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    with torch.no_grad():
        ax.bar(torch.arange(max_number), np.array(m(a[0][exmaple, :].reshape(1, 50)))[0][:max_number], alpha=0.5,
               color='red')
        ax.bar(torch.arange(max_number), yb[exmaple, :max_number], alpha=0.5, color='blue')
    #     plt.plot(torch.arange(10),feed_forword(x_test[0,:].reshape(1,2), linear1)[0][:10])
    plt.savefig('mm1.png')







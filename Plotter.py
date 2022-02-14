# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Carefull, this code asumes everyhting is saved into the training data folder, including the networks!!
# pycharm-community <- Run from terminal if pycharm does not find cuda library

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import os
import math

plottitle = True
titlefont = 12
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=20)


def plot_all_srl(states, rews, data_dict, loss_history, folder='training_results/', trainingcycle=1, episode=1, state_size=5, size=6000, save=True, name=""):
    # Plot newest state representation by SRL network
    closeAll()

    states_pca_2 = getPCA(np.array(states), 2)
    positions = np.array(data_dict["position_obs"])[-len(states):]
    rewards = np.array(rews)
    timesteps = np.array(range(len(positions)))

    plot_priors_history(np.array(loss_history), name='Loss history SRL network')
    
    try:
        plot_2d_colormap(xy=positions, val=timesteps,
                             name="True Agent Position vs Time", label="Timesteps",
                             color="YlGn", plotgoal=True, clip_val=False)
    except:
        print('Could not plot ground truth data')

    plot_2d_colormap(xy=np.array(data_dict["position_obs"]), val=np.array(data_dict["rewards"], dtype=np.float32),
                             name="True Agent Position vs Reward", label="Reward",
                             plotgoal=False, clip_val=True)
    
    try:
        plot_2d_colormap(xy=np.array(data_dict["position_obs"]), val=np.array(data_dict["qvalue"], dtype=np.float32),
                             name="True Agent Position vs Q-value", label="Q-value",
                             plotgoal=False, clip_val=False)
    except:
        print('No q-value found in the data dictionary')

    # plot first 2 principal components of the learned state representation
    plot_2d_colormap(xy=states_pca_2[-size:], val=rewards[-size:],
                             name="First Two Principal Components After Training",
                             label="Reward", clip_val=True)

    # compute and plot PCA
    pca_5 = PCA(n_components=state_size)
    pca_5.fit(states)
    var_explained = pca_5.explained_variance_ratio_ * 100
    plot_bar(values=var_explained, name=" Covariance Matrix Eigenvalue Analysis ", xaxis="PCA Component",
                     yaxis="Percentage of Variance Explained")

    if 'episode_reward' in data_dict.keys():
        # plot cumulated reward
        ma_cumulative_reward = moving_average(np.asarray(data_dict["episode_reward"]), 10)
        plot_history(ma_cumulative_reward, name="Moving Average Reward History (10)", yaxis="Reward")

    # compute and plot TSNE
    try:
        states_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(np.array(states[-size:]))
        plot_2d_colormap(xy=states_embedded, val=rewards[-size:], name="TSNE Visualization",
                                label="Reward", clip_val=True)
    except:
        print('could not make the TSNE plot')
        
    if save:
        saveMultipage(folder + name + "_srl_Large_4walls.pdf")


# Auxiliary function for plotting

def closeAll():
    plt.close('all')

def saveMultipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def getPCA(input, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)

def clip_rewards(rewards): # clip reward to have better scaling of the colors in the colorbar
    rewards[rewards > 0] = 0.3
    rewards[rewards < -2] = -2.1
    plt.pause(0.1)

    return rewards

def plot_2d_colormap(xy, val, name, label, color='magma', plotgoal = False, clip_val=False):


    if clip_val:
        val = clip_rewards(val)

    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)
    im = ax1.scatter(xy[:, 0], xy[:, 1], s=9, c=val, cmap=color, linewidths=0.0)
    if plotgoal:
        ax1.scatter(xy[:,3], xy[:, 4], s=10, c="red", linewidths= 0.01)
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')

    fig.colorbar(im, label= label, ax = ax1)
    plt.pause(0.1)

def plot_trajectories(trajectories, name, legend, plotgoal):

    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)

    for trajectory in trajectories:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], linewidth =1)

    if plotgoal:
        ax1.scatter(trajectories[0][0, 3], trajectories[0][0, 4], s=100, color="red", linewidths=1)

    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.legend(legend)
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.pause(0.1)



def plot_history(hist, name, yaxis, xaxis = "Episode" ):
    timesteps = np.array(range(len(hist)))

    # Visualize loss history
    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, hist, linewidth=1)
    plt.grid()
    # plt.legend(legend)
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)

def plot_bar(values, name, xaxis, yaxis):
    timesteps = range(len(values))

    # plt.ion()
    plt.figure(name)
    plt.bar(timesteps, values)
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)

def plot_mean_dev(mean, dev, name, xaxis, yaxis):
    timesteps = range(len(mean))

    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, mean, 'k', color='#FF0000')
    plt.fill_between(timesteps, mean - dev, mean + dev, alpha=1, edgecolor='#fda3a3', facecolor='#fda3a3', linewidth=0)
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)


def moving_average(data_set, periods=5, multipleaxis=False):
    weights = np.ones(periods) / periods
    if multipleaxis:
        result = []
        for i in range(len(data_set)):
            result.append(np.convolve(data_set[i], weights, mode='valid'))
    else:
        result = np.convolve(data_set, weights, mode='valid')
    return result


def plot_priors_history(epoch_loss_vector, name = 'Loss history SRL network',  folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[: ,0] ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,1] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,2] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,3] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,4] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,5] ,'--' ,linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Temp coherence loss', 'Causality loss', 'Proportionality loss', 'Repeatability loss',
                'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "Priors_losses.pdf"))







# Additional funcition for plotting the Auto-Encoder losses
def plot_AE_history(epoch_loss_vector, name = 'Loss history SRL network',  folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[: ,0] ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,1] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,2] ,'--' ,linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[: ,3] ,'--' ,linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Camera Reconstruction loss', 'Laser Reconstruction loss', 'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize':titlefont})
    plt.ylim((0, 3))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "AE_losses.pdf"))
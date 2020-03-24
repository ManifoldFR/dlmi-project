import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(save_perf, epochs, validate_every, save = False, name= "plot_loss.png", root = "figures"):
    ''' Takes as input the dictionnary defined in interrater_train and plots training and validation loss
    Inputs : 
        save_perf : performance dictionnary
        epochs : nb of epochs, int
        validate_every : int
        save : whether to save the plot
    '''
    plt.scatter(np.array(range(epochs))+1, save_perf["train_loss"],label="train")
    plt.scatter(np.array(range(0,epochs,validate_every))+1, save_perf["val_loss"],label="validation")
    plt.xticks(range(epochs+1))
    plt.legend()
    plt.title("Train and validation loss after "+str(epochs)+" epochs")
    if save == True : 
        plt.savefig(os.path.join("interrater", root, str(name+".png")))
    plt.show()
        

def plot_metrics(metric, save_perf, epochs, validate_every, save = False, name= "plot_metrics.png", root = "figures"):
    plt.scatter(np.array(range(epochs))+1, [save_perf["train_acc"][i][metric] for i in range(epochs)],label="train")
    plt.scatter(np.array(range(0,epochs,validate_every))+1, [save_perf["val_acc"][i][metric] for i in range(0, epochs, validate_every)],label="validation")
    plt.xticks(range(epochs+1))
    plt.legend()
    plt.title("Train and validation "+metric+ " after "+str(epochs)+" epochs")
    if save == True : 
        plt.savefig(os.path.join("interrater", root, str(name+"_"+metric+".png")))
    plt.show()














import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(save_perf, epochs, validate_every, start_at_epoch = 1, save = False, name= "plot_loss", root = "figures"):
    ''' Takes as input the dictionnary defined in interrater_train and plots training and validation loss
    Inputs : 
        save_perf : performance dictionnary
        epochs : nb of epochs, int
        validate_every : int
        save : whether to save the plot
    '''
    assert start_at_epoch <= epochs
    plt.scatter(np.array(range(start_at_epoch-1, epochs))+1, save_perf["train_loss"][start_at_epoch-1:],label="train")
    plt.scatter(np.array(range(start_at_epoch-1, epochs, validate_every))+1, save_perf["val_loss"][start_at_epoch-1:],label="validation")
    plt.xticks(range(start_at_epoch,epochs+1))
    plt.axis([start_at_epoch-1 + 0.5, 
              epochs+1.5, 
              np.min(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:])-(np.max(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:])-np.min(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:]))*0.2, 
              np.max(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:])+(np.max(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:])-np.min(save_perf["val_loss"][start_at_epoch-1:]+save_perf["train_loss"][start_at_epoch-1:]))*0.2])
    plt.legend()
    plt.title("Train and validation loss after "+str(epochs)+" epochs")
    if save == True : 
        plt.savefig(os.path.join("interrater", root, str(name+".png")))
    plt.show()
        

def plot_metrics(metric, save_perf, epochs, validate_every, start_at_epoch = 1, save = False, name= "plot_metrics", root = "figures"):
    assert start_at_epoch <= epochs
    train_perf = [save_perf["train_acc"][i][metric] for i in range(start_at_epoch-1, epochs)]
    val_perf = [save_perf["val_acc"][i][metric] for i in range(start_at_epoch-1, epochs, validate_every)]
    plt.scatter(np.array(range(start_at_epoch-1, epochs))+1, train_perf,label="train")
    plt.scatter(np.array(range(start_at_epoch-1 ,epochs,validate_every))+1, val_perf,label="validation")
    plt.xticks(range(start_at_epoch,epochs+1))
    plt.axis([start_at_epoch-1 + 0.5, 
              epochs+1.5, 
              np.min(val_perf+train_perf)-(np.max(val_perf+train_perf)-np.min(val_perf+train_perf))*0.2, 
              np.max(val_perf+train_perf)+(np.max(val_perf+train_perf)-np.min(val_perf+train_perf))*0.2])
    plt.legend()
    plt.title("Train and validation "+metric+ " after "+str(epochs)+" epochs")
    if save == True : 
        plt.savefig(os.path.join("interrater", root, str(name+"_"+metric+".png")))
    plt.show()




def plot_target_output(save_perf, metric = "IoU", save = False, name= "plot_target_output", root = "figures"):
    nb_epochs = len(save_perf["val_details"])
    nb_val = len(save_perf["val_details"][nb_epochs-1]["outputs"])
    output = save_perf["val_details"][nb_epochs-1]["outputs"]
    target = save_perf["val_details"][nb_epochs-1]["targets"]
    
    plt.scatter(np.array(range(nb_val))+1, output, label="outputs")
    plt.scatter(np.array(range(nb_val))+1, target, label="targets")
    plt.xticks(np.array(range(nb_val))+1)
    plt.axis([0.5, 
              nb_val+1.5, 
              np.min(output+target)-(np.max(output+target)-np.min(output+target))*0.2, 
              np.max(output+target)+(np.max(output+target)-np.min(output+target))*0.2])
    plt.legend()
    plt.title("Output VS target " + metric+" after "+str(nb_epochs)+" epochs")
    if save == True : 
        plt.savefig(os.path.join("interrater", root, str(name+"_"+metric+".png")))
    plt.show()

















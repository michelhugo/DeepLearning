# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:16:00 2019

@author: Georges
"""
import matplotlib.pyplot as plt
import pickle

def __main__():
    # Get accuracy

    with open('accuracy_ours_v2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        acc_h = pickle.load(f)
    
    with open('accuracy_pytorch_v2.pkl', 'rb') as g:  # Python 3: open(..., 'rb')
        accuracies_ = pickle.load(g)
        
    with open('loss_ours_v2.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        loss_h = pickle.load(f)
    
    with open('loss_pytorch_v2.pkl', 'rb') as g:  # Python 3: open(..., 'rb')
        losses_ = pickle.load(g)
        
    final_fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)
    
    ax1 = plt.subplot(2,1,1)
    ax2 = ax1.twinx()
    
    f1 = ax1.plot(acc_h[:60] * 100, color='darkblue', label='Our Framework')
    
    f2 = ax2.plot(accuracies_[:60], color='crimson', label='PyTorch')
    
    fs = f1 + f2
    labs = [l.get_label() for l in fs]
    ax1.legend(f1+f2, labs, loc='lower right', fontsize=15)
    
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Accuracy [%]', fontsize=15)
    #ax2.set_ylabel('Accuracy [%]', fontsize=15)
    
    plt.title('Accuracy evolution during training', fontsize=20)
    
    ax1 = plt.subplot(2,1,2)
    ax2 = ax1.twinx()
    
    f1 = ax1.plot(loss_h[:150], color='darkblue', label='Our Framework')
    
    f2 = ax2.plot(losses_[:150], color='crimson', label='PyTorch')
    
    fs = f1 + f2
    labs = [l.get_label() for l in fs]
    ax1.legend(f1+f2, labs, loc='center right', fontsize=15)
    
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('MSE loss', fontsize=15)
    ax2.set_ylabel('MSE loss Pytorch', fontsize=15)
    
    plt.title('Loss evolution during training', fontsize=20)
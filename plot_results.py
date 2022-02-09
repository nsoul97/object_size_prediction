import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
import numpy as np
from config import MOV_COMPLETION, FS_NAMES

fig1, axes1 = None, None
fig2, axes2 = None, None

FS_MOV = 0
METHOD_MOV = 0

def close(event):
    """ If the user closes one of the figure windows, the program is terminated.

    Args:
        event (matplotlib.backend_bases.CloseEvent): The event that was triggered by a figure being closed.
    """

    exit(0)

def press_key(event):
    """ Identify which key the user pressed while one of the figures was the active window.
        The 'up' and 'down' arrow keys are used to change the feature set for which the figures are plotted.
        The 'right' and 'left' arrow keys are used to change the model for which the figures are plotted.
        The 'esc' key is used to terminate the program.

    Args:
        event (matplotlib.backend_bases.KeyEvent): The event that was triggered by pressing a key when one of the figures was the active window.
    """

    global FS_MOV, METHOD_MOV
    if event.key == 'up':
        FS_MOV = 1
    elif event.key == 'down':
        FS_MOV = -1
    elif event.key == 'right':
        METHOD_MOV = 1
    elif event.key == 'left':
        METHOD_MOV = -1
    elif event.key == 'escape':
        exit(0)
    else:
        FS_MOV = 0
        METHOD_MOV = 0
    
    fig1.canvas.stop_event_loop()

def update_featimp_plot(feat_imp_dict, strategy, fsid):
    """ The feature imporances of the ExtraTrees model are plotted in the same figure for the given dataset split strategy, the given feature set and for each of the 20%, 40%, 60%, 80% and 100% movement completion 
        percentages. 

    Args:
        feat_imp_dict (dictionary): A 2-level dictionary with a feature set id as the 1st-level key and the movement completion percentage as the 2nd-level key. The value of the dictionary in the 2nd level is a tuple, whose
                                    1st value is a numpy array which indicates the importance of the features for the given feature set and movement completion percentage. The feature importance values were calculated as the average
                                    of k feature importance vectors of the ExtraTrees model in the k-fold evaluation scheme. The 2nd value of the tuple is also a numpy array, which indicates the standard deviation of the k feature
                                    importance vectors.
        strategy (string): The dataset split strategy string ('all-in', 'one-out') for which the accuracies are plotted.
        fsid (int): An integer that is used to identify the feature set for which the accuracies are plotted. ([0,8]).
    """

    DESCR_FEATURES = ["number of frames", "minimum", "maximum", "mean", "standard deviation", "slope"]
    tot_kv = len(FS_NAMES[fsid])
    tot_kdf = len(DESCR_FEATURES)-1

    max_val = np.max([np.max(feat_imp_dict[fsid][t][0]) for t in MOV_COMPLETION])

    fig3.suptitle(f"Dataset split strategy: '{strategy}'\nFeature set: {FS_NAMES[fsid]}")
    for k, t in enumerate(MOV_COMPLETION):
        clr = [[feat_imp_dict[fsid][t][0][0] for _ in range(tot_kv)]] + [[feat_imp_dict[fsid][t][0][1+i+j*tot_kdf] for j in range(tot_kv)] for i in range(tot_kdf)]

        axes3[k].cla()
        if k:
            sn.heatmap(clr, cmap='Oranges', cbar=False, xticklabels=FS_NAMES[fsid], yticklabels=False, ax=axes3[k], vmin=0, vmax=max_val)
        else:
            sn.heatmap(clr, cmap='Oranges', cbar=False, xticklabels=FS_NAMES[fsid], yticklabels=DESCR_FEATURES, ax=axes3[k], vmin=0, vmax=max_val)
                    
        axes3[k].set_title(f"{MOV_COMPLETION[k]}%", fontsize=16)
        axes3[k].tick_params(left=False, bottom=False)
        axes3[k].set_yticklabels(axes3[k].get_yticklabels(), fontsize = 14)
        axes3[k].set_xticklabels(axes3[k].get_xticklabels(), rotation = 90, fontsize = 14)



def update_accuracies_plot(acc_dict, strategy, fsid, methods):
    """ The average accuracies of all the given models are plotted in the same figure for the given dataset split strategy, the given feature set and for each of the 20%, 40%, 60%, 80% and 100% movement completion
        percentages.

    Args:
        acc_dict (dictionary): A 3-level dictionary with a feature set id as the 1st-level key, a movement completion percentage as the 2nd-level key and a method as the 3rd-level key. The value of the dictionary in the
                               3rd level is a tuple, whose 1st value indicates the accuracy of the method for the given feature set and movement completion percentage. This accuracy was calculated as
                               the average of k accuracy results in the k-fold evaluation scheme. The 2nd value of the tuple is the standard deviation of the k accuracy results. 
        strategy (string): The dataset split strategy string ('all-in', 'one-out') for which the accuracies are plotted.
        fsid (int): An integer that is used to identify the feature set for which the accuracies are plotted. ([0,8])
        methods (list): A list of string containing the names of all the models that were evaluated. ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess')
    """

    colors = cm.rainbow((np.linspace(0,1,len(methods))))
    axes1.cla()
    fig1.suptitle(f"Dataset split strategy: '{strategy}'\nFeature set: {FS_NAMES[fsid]}")
    axes1.set_xlabel('Movement Ratio (%)')
    axes1.set_ylabel('Accuracy (%)')
    axes1.set_xticks(MOV_COMPLETION)

    for i, method in enumerate(methods):
        axes1.plot(MOV_COMPLETION, [acc_dict[fsid][t][method][0]*100 for t in MOV_COMPLETION], c=colors[i], label=method)
    axes1.legend(loc='upper left')


def update_confmtx_plot(conf_mtx_dict, strategy, fsid, method):
    """ The confusion matrices of the given model are plotted in the same figure for the given dataset split strategy, the given feature set and for each of the 20%, 40%, 60%, 80% and 100% movement completion
        percentages.

    Args:
        conf_mtx_dict (dictionary): A 3-level dictionary with a feature set id as the 1st-level key, a movement completion percentage as the 2nd-level key and a method as the 3rd-level key. The value of the dictionary in the
                                    3rd level is a tuple, whose 1st value is a 3x3 numpy array which indicates the confusion matrix of the given feature set and movement completion percentage. This confusion matrix was
                                    calculated as the average of k confusion matrices in the k-fold evaluation scheme. The 2nd value of the tuple is also a 3x3 numpy array, which indicates the standard deviation of the k
                                    confusion matrices. 
        strategy (string): The dataset split strategy string ('all-in', 'one-out') for which the confusion matrices are plotted.
        fsid (int): An integer that is used to identify the feature set for which the confusion matrices are plotted. ([0,8])
        method (string): A string that identifies the model for which the confusion matrices are plotted. ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess')
    """

    LABELS = ['S', 'M', 'L']
    
    fig2.suptitle(f"Dataset split strategy: '{strategy}'\nFeature set: {FS_NAMES[fsid]}\nModel: {method}")
    for k, t in enumerate(MOV_COMPLETION):
        axes2[k].cla()
        annotations = [[f"{100*conf_mtx_dict[fsid][t][method][0][i,j]:.1f}\n({100*conf_mtx_dict[fsid][t][method][1][i,j]:.1f})" for j in range(3)] for i in range(3)]
        if k:
            sn.heatmap(conf_mtx_dict[fsid][t][method][0], cmap='Oranges', annot=annotations, fmt="", linecolor='w', linewidths=0.5, annot_kws={"size":18}, cbar=False, xticklabels=LABELS, yticklabels=False, ax=axes2[k], vmin=0.0, vmax=1.0)
        else:
            sn.heatmap(conf_mtx_dict[fsid][t][method][0], cmap='Oranges', annot=annotations, fmt="", linecolor='w', linewidths=0.5, annot_kws={"size":18}, cbar=False, xticklabels=LABELS, yticklabels=LABELS, ax=axes2[k], vmin=0.0, vmax=1.0)
            axes2[k].set_yticklabels(axes2[k].get_yticklabels(), rotation = 0, fontsize = 14)
                    
        axes2[k].set_xticklabels(axes2[k].get_xticklabels(), rotation = 0, fontsize = 14)
        axes2[k].set_title(f"{MOV_COMPLETION[k]}%", fontsize=16)
        axes2[k].tick_params(left=False, bottom=False)


        axes2[0].set_ylabel('Ground Truth', fontsize=16)
        axes2[2].set_xlabel('Predictions', fontsize=16)

def plot_results(acc_dict, conf_mtx_dict, feat_imp_dict, strategy, fs_ids, methods):
    """ Plot the accuracies and the confusion matrices interactively.
        The user can select the feature set and the model name for which the results are plotted. 

    Args:
        acc_dict (dictionary): A 3-level dictionary with a feature set id as the 1st-level key, a movement completion percentage as the 2nd-level key and a method as the 3rd-level key. The value of the dictionary in the
                               3rd level is a tuple, whose 1st value indicates the accuracy of the method for the given feature set and movement completion percentage. This accuracy was calculated as
                               the average of k accuracy results in the k-fold evaluation scheme. The 2nd value of the tuple is the standard deviation of the k accuracy results. 
        conf_mtx_dict (dictionary): A 3-level dictionary with a feature set id as the 1st-level key, a movement completion percentage as the 2nd-level key and a method as the 3rd-level key. The value of the dictionary in the
                                3rd level is a tuple, whose 1st value is a 3x3 numpy array which indicates the confusion matrix of the given feature set and movement completion percentage. This confusion matrix was
                                calculated as the average of k confusion matrices in the k-fold evaluation scheme. The 2nd value of the tuple is also a 3x3 numpy array, which indicates the standard deviation of the k
                                confusion matrices.
        feat_imp_dict (dictionary): A 2-level dictionary with a feature set id as the 1st-level key and the movement completion percentage as the 2nd-level key. The value of the dictionary in the 2nd level is a tuple, whose
                                    1st value is a numpy array which indicates the importance of the features for the given feature set and movement completion percentage. The feature importance values were calculated as the average
                                    of k feature importance vectors of the ExtraTrees model in the k-fold evaluation scheme. The 2nd value of the tuple is also a numpy array, which indicates the standard deviation of the k feature
                                    importance vectors. If an ExtraTrees model was not used, the parameter has a None value.
        strategy (string): The dataset split strategy string ('all-in', 'one-out') for which the models were evaluated.
        fs_ids (list): A list of integers containing the feature set ids that were evaluated. ([0,8])
        methods (list): A list of string containing the names of all the models that were evaluated. ('RandomForest', 'GradientBoosting', 'ExtraTrees', 'SVM', 'GaussianProcess')
    """

    global FS_MOV, METHOD_MOV, fig1, axes1, fig2, axes2, fig3, axes3

    plt.ion()
    fig1, axes1 = plt.subplots(1,1, tight_layout=True)
    fig2, axes2 = plt.subplots(1,5, tight_layout=True)
    for fig in [fig1, fig2]:
        fig.canvas.mpl_connect('key_press_event', press_key)
        fig.canvas.mpl_connect('close_event', close)

    if feat_imp_dict:
        fig3, axes3 = plt.subplots(1, 5)
        fig3.canvas.mpl_connect('key_press_event', press_key)
        fig3.canvas.mpl_connect('close_event', close)

    fs_ids, methods = sorted(fs_ids), sorted(methods)

    fsid_ind = 0
    fsid = fs_ids[fsid_ind]
    update_accuracies_plot(acc_dict, strategy, fsid, methods)
    if feat_imp_dict:
        update_featimp_plot(feat_imp_dict, strategy, fsid)

    method_ind = 0
    method = methods[method_ind]
    update_confmtx_plot(conf_mtx_dict, strategy, fsid, method)
    
    while True:        
        if FS_MOV:
            if (fsid_ind == 0 and FS_MOV == 1) or (fsid_ind == len(fs_ids)-1 and FS_MOV == -1) or (fsid_ind > 0 and fsid_ind < len(fs_ids)-1):
                fsid_ind = (fsid_ind + FS_MOV) % len(fs_ids)
                fsid = fs_ids[fsid_ind]
                update_accuracies_plot(acc_dict, strategy, fsid, methods)
                method = methods[method_ind]
                update_confmtx_plot(conf_mtx_dict, strategy, fsid, method)
                if feat_imp_dict:
                    update_featimp_plot(feat_imp_dict, strategy, fsid)

        if METHOD_MOV:
            if (method_ind == 0 and METHOD_MOV == 1) or (method_ind == len(methods)-1 and METHOD_MOV == -1) or (method_ind > 0 and method_ind < len(methods)-1): 
                method_ind = (method_ind + METHOD_MOV) % len(methods)
                method = methods[method_ind]
                fsid = fs_ids[fsid_ind]
                update_confmtx_plot(conf_mtx_dict, strategy, fsid, method)

        FS_MOV = 0
        METHOD_MOV = 0

        fig1.canvas.start_event_loop()
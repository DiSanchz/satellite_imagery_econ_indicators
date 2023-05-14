from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#######################################################
# PERFORMANCE METRICS
#######################################################

def plot_acc_loss(model):
     ##Plot for the accuracy of the baseline model 
    accuracy_train = model.history['accuracy']
    accuracy_val = model.history['val_accuracy']
    plt.plot(accuracy_train, label='training_accuracy')
    plt.plot(accuracy_val, label='validation_accuracy')
    plt.title('EVOLUTION OF VALIDATION ACCURACY')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    ##Plot for the loss of the baseline model 
    loss_train = model.history['loss']
    loss_val = model.history['val_loss']
    plt.plot(loss_train, label='training_loss')
    plt.plot(loss_val, label='validation_loss')
    plt.title('EVOLUTION OF LOSS')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return 

def plot_ROC(y_predict,y_test,num_clas): 

    fpr = {}
    tpr = {}
    roc_auc = {}
    #calculating roc for each class
    for i in range(num_clas):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_predict[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
  
  # calculating micro-average ROC curve and  area
    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc_micro = roc_auc_score(y_test.ravel(), y_predict.ravel())

  # Compute macro-average ROC curve and  area
    fpr_macro = np.unique(np.concatenate([fpr[i] for i in range(num_clas)]))
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(num_clas):
        tpr_macro += np.interp(fpr_macro, fpr[i], tpr[i])
    tpr_macro /= num_clas
    roc_auc_macro = auc(fpr_macro, tpr_macro)

  #Plot the ROC curve for each class using matplotlib.pyplot.plot()
    plt.figure(figsize=(10, 5))
    lw = 2
    for i in range(num_clas):
        plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class %d (area = %0.2f)' % (i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr_micro, tpr_micro,lw=lw, linestyle='--', label='micro-average ROC curve (area = %0.2f)' % (roc_auc_micro))
    plt.plot(fpr_macro, tpr_macro,lw=lw, linestyle='--', label='macro-average ROC curve (area = %0.2f)' % (roc_auc_macro))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of Multiclass')
    plt.legend(loc="lower right")
    plt.show()
    return 

def plot_cm(y_test,y_pred):

    predictions = np.argmax(y_pred, axis=1) 
    true_labels = np.argmax(y_test, axis=1)
    labels = ['Quartile 1', 'Quartile 2', 'Quartile 3', 'Quartile 4']

    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d',  xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
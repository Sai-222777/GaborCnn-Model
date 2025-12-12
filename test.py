import torch
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(conf_matrix, classes, cmap=plt.cm.Blues, title='Confusion Matrix (%)', filename='confusion_matrix.png'):
    """
    Plots a confusion matrix as a heatmap with percentage values (no % symbol).
    Expects conf_matrix in percentages (0-100).
    Saves the plot to a file.

    Parameters:
    - filename: The path to save the plot (default is 'confusion_matrix.png').
    """
    plt.figure(figsize=(8, 6))
    # Format annotations to show percentages with 2 decimal places but no % symbol
    annot = np.array([[f"{val:.2f}" for val in row] for row in conf_matrix])
    sns.heatmap(conf_matrix, annot=annot, fmt='', cmap=cmap, 
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    
    # Save the plot to the specified filename
    plt.savefig(filename)
    plt.close()  # Close the figure to avoid showing it

def test_model(model, test_batches, criterion, device="cuda", num_classes=4, class_names=None):
    model.eval()
    all_outputs = []
    all_targets = []
    total_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in tqdm(test_batches, desc="Testing"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            all_outputs.append(outputs)
            all_targets.append(y)
            
            _, preds = torch.max(outputs, dim=1)
            for t, p in zip(y.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    end_time = time.time()
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    _, preds = torch.max(all_outputs, dim=1)
    correct = (preds == all_targets).sum().item()
    total = all_targets.size(0)
    avg_loss = total_loss / len(test_batches)
    
    sensitivities = []
    specificities = []

    accuracies = []
    
    for i in range(num_classes):
        TP = confusion_matrix[i, i].item()
        FN = confusion_matrix[i].sum().item() - TP
        FP = confusion_matrix[:, i].sum().item() - TP
        TN = confusion_matrix.sum().item() - (TP + FN + FP)
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        accuracies.append(accuracy)
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    avg_sensitivity = sum(sensitivities) / num_classes
    avg_specificity = sum(specificities) / num_classes
    
    all_probs = torch.softmax(all_outputs, dim=1).cpu().numpy()
    all_targets_np = all_targets.cpu().numpy()
    
    auc_scores = []
    for i in range(num_classes):
        binary_true = (all_targets_np == i).astype(int)
        auc = roc_auc_score(binary_true, all_probs[:, i])
        auc_scores.append(auc)
    avg_auc = np.mean(auc_scores)
    
    print(f"⏱️ Inference time: {end_time - start_time:.4f} seconds")
    print(f"✅ Test Accuracy: {100 * correct / total:.2f}% | Loss: {avg_loss:.4f}")
    
    for i in range(num_classes):
        print(f"Class {i} - Sensitivity: {sensitivities[i]:.4f}, Specificity: {specificities[i]:.4f}")
        
    print(f"🌍 Average Sensitivity: {avg_sensitivity:.4f}")
    print(f"🌍 Average Specificity: {avg_specificity:.4f}")
    
    for i in range(num_classes):
        print(f"Class {i} - AUC: {auc_scores[i]:.4f}")
        
    print(f"🌍 Average AUC: {avg_auc:.4f}")

    # print()
    for i in range(num_classes):
        print(f"Class {i} - Accuracy: {accuracies[i]:.4f}")
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Normalize confusion matrix row-wise to percentages
    confusion_matrix_percent = confusion_matrix / confusion_matrix.sum(dim=1, keepdim=True) * 100
    confusion_matrix_percent = confusion_matrix_percent.numpy()
    
    plot_confusion_matrix(confusion_matrix_percent, classes=class_names)

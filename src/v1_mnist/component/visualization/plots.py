import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid")

def plot_class_distribution(counts_dict, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    classes = list(counts_dict.keys())
    counts = [d['count'] for d in counts_dict.values()]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=classes, y=counts, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(classes)
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
                   
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, classes, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sample_images(dataset, num_samples_per_class, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_classes = 10
    fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(num_samples_per_class*1.5, num_classes*1.5))
    
    class_counts = {i: 0 for i in range(num_classes)}
    
    for img, label in dataset:
        label = label.item()
        if class_counts[label] < num_samples_per_class:
            ax = axes[label, class_counts[label]]
            # Convert from [1, 28, 28] tensor to [28, 28] numpy array
            img_np = img.squeeze().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')
            if class_counts[label] == 0:
                ax.set_title(f"Class {label}", loc='left')
            class_counts[label] += 1
            
        if all(count == num_samples_per_class for count in class_counts.values()):
            break
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

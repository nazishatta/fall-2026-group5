import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = np.mean(all_preds == all_labels)
    clf_report_str = classification_report(all_labels, all_preds, zero_division=0)
    clf_report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': acc,
        'classification_report_str': clf_report_str,
        'classification_report_dict': clf_report_dict,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def print_evaluation_summary(eval_results, split_name):
    print(f"\n--- Evaluation Summary: {split_name.upper()} ---")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(eval_results['classification_report_str'])

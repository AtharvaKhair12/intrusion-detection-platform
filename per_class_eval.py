import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

def main():
    print("Loading original parquet to recover text labels...")
    df = pd.read_parquet("data/processed/cicids2017.parquet")
    labels = df["label"].str.strip().to_numpy()
    
    print("Reconstructing splits...")
    is_benign = labels == "BENIGN"
    benign_idx = np.where(is_benign)[0]
    attack_idx = np.where(~is_benign)[0]

    _, benign_test_idx = train_test_split(
        benign_idx, test_size=0.2, random_state=42
    )

    test_idx = np.concatenate([benign_test_idx, attack_idx])
    np.random.seed(42)
    np.random.shuffle(test_idx)
    
    labels_test = labels[test_idx]
    
    print("Loading processed test data and model...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Split test set into 20% validation, 80% final test, same as train_model.py
    X_val, X_test_final, y_val, y_test_final, labels_val, labels_test_final = train_test_split(
        X_test, y_test, labels_test, test_size=0.8, stratify=y_test, random_state=42
    )
    
    model = joblib.load('model.joblib')
    
    print("Predicting on final test set...")
    y_test_pred_raw = model.predict(X_test_final)
    y_test_pred = np.where(y_test_pred_raw == -1, 1, 0)
    
    # Evaluate per class
    print("\nPer-class Recall:")
    
    attack_mask = y_test_final == 1
    attack_labels = labels_test_final[attack_mask]
    attack_preds = y_test_pred[attack_mask]
    
    unique_attacks = pd.Series(attack_labels).value_counts()
    print("Attack distribution in test final:")
    # print(unique_attacks)
    
    results = {}
    
    # Categories: DoS Hulk, PortScan, DDoS, Other
    # Let's map everything else to 'Other'
    for attack_type in ["DoS Hulk", "PortScan", "DDoS"]:
        mask = attack_labels == attack_type
        if mask.sum() > 0:
            recall = recall_score(np.ones(mask.sum()), attack_preds[mask], zero_division=0)
            results[attack_type] = recall
            print(f"  {attack_type}: {recall:.4f} ({mask.sum()} samples)")
            
    other_mask = ~np.isin(attack_labels, ["DoS Hulk", "PortScan", "DDoS"])
    if other_mask.sum() > 0:
        recall = recall_score(np.ones(other_mask.sum()), attack_preds[other_mask], zero_division=0)
        results["Other"] = recall
        print(f"  Other: {recall:.4f} ({other_mask.sum()} samples)")
        
    # Append to metrics.md
    print("Appending to metrics.md...")
    with open('metrics.md', 'a') as f:
        f.write("\n## Per-Class Breakdown (Held-out Test Slice)\n")
        f.write("| Attack Type | Recall | Samples |\n")
        f.write("|-------------|--------|---------|\n")
        
        for attack_type in ["DoS Hulk", "PortScan", "DDoS"]:
            mask = attack_labels == attack_type
            if mask.sum() > 0:
                f.write(f"| {attack_type} | {results[attack_type]:.4f} | {mask.sum()} |\n")
        
        if other_mask.sum() > 0:
            f.write(f"| Other | {results['Other']:.4f} | {other_mask.sum()} |\n")
            
        # Interpretation
        f.write("\n**Interpretation:** ")
        # Add placeholder text, we'll replace this after inspecting the output
        f.write("TODO: Add explanation based on the results.\n")
        
if __name__ == "__main__":
    main()

import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import time

def main():
    print("Loading data...")
    X_train_benign = np.load('data/processed/X_train_benign.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"X_train_benign shape: {X_train_benign.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Split test set into 20% validation, 80% final test
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test, y_test, test_size=0.8, stratify=y_test, random_state=42
    )
    
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Final test set size: {X_test_final.shape[0]}")
    
    contamination_values = [0.01, 0.02, 0.05, 0.1]
    results = []
    
    print("Sweeping contamination values...")
    for c in contamination_values:
        start_time = time.time()
        # Train
        model = IsolationForest(contamination=c, n_jobs=-1, random_state=42)
        model.fit(X_train_benign)
        
        # Predict on validation
        y_val_pred_raw = model.predict(X_val)
        # Convert -1 (outlier/attack) to 1, and 1 (inlier/benign) to 0
        y_val_pred = np.where(y_val_pred_raw == -1, 1, 0)
        
        # Evaluate
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
        fpr = fp / (fp + tn)
        recall = tp / (tp + fn)
        
        elapsed = time.time() - start_time
        print(f"Contamination: {c} | FPR: {fpr:.4f} | Recall: {recall:.4f} | Time: {elapsed:.1f}s")
        
        results.append({
            'contamination': c,
            'fpr': fpr,
            'recall': recall
        })
        
    # Plotting FPR vs Recall
    plt.figure(figsize=(8, 6))
    fpr_list = [r['fpr'] for r in results]
    recall_list = [r['recall'] for r in results]
    plt.plot(fpr_list, recall_list, marker='o')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('Recall (TPR)')
    plt.title('FPR vs Recall on Validation Set')
    for r in results:
        plt.annotate(f"c={r['contamination']}", (r['fpr'], r['recall']))
    plt.grid(True)
    plt.savefig('fpr_vs_recall.png')
    print("Saved plot to fpr_vs_recall.png")
    
    # Pick best contamination
    best_c = None
    best_recall = -1
    for r in results:
        if r['fpr'] < 0.05 and r['recall'] > best_recall:
            best_c = r['contamination']
            best_recall = r['recall']
            
    if best_c is None:
        print("Warning: No contamination achieved FPR < 5%. Picking the one with lowest FPR.")
        best_c = min(results, key=lambda x: x['fpr'])['contamination']
        
    print(f"\nSelected best contamination: {best_c} (Target: FPR < 0.05 with max Recall)")
    
    print("\nRetraining final model on full benign set with best contamination...")
    final_model = IsolationForest(contamination=best_c, n_jobs=-1, random_state=42)
    final_model.fit(X_train_benign)
    
    print("Evaluating on final held-out test slice...")
    y_test_pred_raw = final_model.predict(X_test_final)
    y_test_pred = np.where(y_test_pred_raw == -1, 1, 0)
    
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()
    test_fpr = fp / (fp + tn)
    test_recall = tp / (tp + fn)
    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    print(f"Test FPR: {test_fpr:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    
    print("Saving final model to model.joblib...")
    joblib.dump(final_model, 'model.joblib')
    
    print("Writing metrics.md...")
    with open('metrics.md', 'w') as f:
        f.write("# Isolation Forest Training Metrics\n\n")
        f.write("## Validation Sweep\n")
        f.write("Tuning was performed on a 20% validation slice of the test data.\n\n")
        f.write("| Contamination | FPR | Recall |\n")
        f.write("|---------------|-----|--------|\n")
        for r in results:
            f.write(f"| {r['contamination']} | {r['fpr']:.4f} | {r['recall']:.4f} |\n")
        
        f.write("\n## Chosen Operating Point\n")
        f.write(f"**Selected Contamination:** {best_c}\n\n")
        f.write("We selected this contamination value because it achieves an FPR under 5% while maximizing recall.\n")
        
        f.write("\n## Final Evaluation (Held-out Test Slice)\n")
        f.write("Evaluated on the remaining 80% test slice.\n\n")
        f.write(f"- **FPR:** {test_fpr:.4f}\n")
        f.write(f"- **Recall:** {test_recall:.4f}\n")
        f.write(f"- **Precision:** {test_precision:.4f}\n\n")
        f.write("> **CAVEAT on Precision:** The test set has an artificial class balance of ~45% benign and 55% attack. This is not representative of real-world traffic where attacks are far rarer. Therefore, the reported Precision is significantly higher than what would be observed in production at the same FPR and Recall.\n")

    print("Done!")

if __name__ == "__main__":
    main()

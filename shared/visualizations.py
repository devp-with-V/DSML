import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display

def display_execution_plots(problem_type, results):
    """
    Takes the results dictionary from a method execution and attempts to render
    standard data science plots inline in the Jupyter Notebook.
    """
    # If the method didn't return y_test or preds, we can't plot much.
    y_test = results.get("y_test")
    preds = results.get("preds")
    
    if preds is None:
        print("No predictions available to plot.")
        return
        
    sns.set_theme(style="whitegrid", palette="muted")
    
    # === Classification Plots ===
    if problem_type == "classification":
        if y_test is None:
            print("y_test is required for classification plots.")
            return
            
        from sklearn.metrics import confusion_matrix
        
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()

    # === Regression Plots ===
    elif problem_type == "regression":
        if y_test is None:
            print("y_test is required for regression plots.")
            return
            
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, preds, alpha=0.5, color='b')
        
        # Plot perfect prediction line
        min_val = min(np.min(y_test), np.min(preds))
        max_val = max(np.max(y_test), np.max(preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted', fontsize=16)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.tight_layout()
        plt.show()

    # === Clustering Plots ===
    elif problem_type == "clustering":
        X_reduced = results.get("X_reduced")
        
        # For clustering, we just plot the preds if we can reduce to 2D
        if X_reduced is not None and X_reduced.shape[1] >= 2:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=preds, palette='viridis', legend='full')
            plt.title('Cluster Visualization (2D Projection)', fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            print("2D projection not provided by model, cannot plot clusters automatically.")
            
    # === Dimensionality Reduction Plots ===
    elif problem_type == "dimensionality_reduction":
        X_reduced = results.get("X_reduced")
        if X_reduced is not None and X_reduced.shape[1] >= 2:
            plt.figure(figsize=(8, 5))
            if y_test is not None:
                sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y_test, palette='coolwarm')
            else:
                sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], alpha=0.7)
            plt.title('Dimensionality Reduction (2D)', fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            print("Output does not have 2 dimensions to plot.")
            
    else:
        print(f"Automated plotting not configured for problem type: {problem_type}")

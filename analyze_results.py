import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for premium visualization
plt.style.use('bmh')
sns.set_palette("viridis")

def analyze_results(csv_path="result.csv"):
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # Clean numeric columns
    numeric_cols = ["Best dice", "Best Epoch", "Train Loss", "Valid Dice", "Valid Iou", "Valid Loss", "Runtime"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter only finished runs for fair performance comparison
    finished_df = df[df["State"] == "finished"].copy()
    
    # 2. Performance by Architecture
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='net_structure', y='Best dice', data=finished_df)
    plt.title("Performance Distribution by Network Structure (Finished Runs)", fontsize=14)
    plt.ylabel("Best Dice Score")
    plt.xlabel("Architecture")
    plt.savefig("arch_performance.png", bbox_inches='tight')
    
    # 3. Backbone Comparison
    # Filter only Fusion models that have backbones
    fusion_df = finished_df[finished_df["net_structure"].str.contains("Fusion", na=False)]
    if not fusion_df.empty:
        plt.figure(figsize=(10, 6))
        # Fill None backbones with 'none' string for plotting
        fusion_df['encoder_backbone'] = fusion_df['encoder_backbone'].fillna('none')
        sns.barplot(x='encoder_backbone', y='Best dice', data=fusion_df, ci='sd')
        plt.title("Impact of Encoder Backbone on Fusion Models", fontsize=14)
        plt.xticks(rotation=45)
        plt.savefig("backbone_impact.png", bbox_inches='tight')
    
    # 4. Correlation Analysis (Metrics)
    plt.figure(figsize=(8, 6))
    corr_cols = ["Best dice", "Valid Iou", "Train Loss", "Valid Loss"]
    sns.heatmap(finished_df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Training Metrics", fontsize=14)
    plt.savefig("metrics_correlation.png", bbox_inches='tight')

    # 5. Training Gap (Overfitting Check)
    plt.figure(figsize=(10, 6))
    plt.scatter(finished_df["Train Loss"], finished_df["Valid Loss"], alpha=0.6, c=finished_df["Best dice"], cmap='viridis')
    plt.colorbar(label='Best Dice')
    plt.xlabel("Train Loss")
    plt.ylabel("Valid Loss")
    plt.title("Train Loss vs Valid Loss (Overfitting Analysis)", fontsize=14)
    plt.savefig("overfitting_analysis.png", bbox_inches='tight')

    # 6. Summary Stats for Commentary
    summary = {
        "best_overall": finished_df.loc[finished_df["Best dice"].idxmax()][["Name", "Best dice", "net_structure", "encoder_backbone"]].to_dict(),
        "avg_dice_by_arch": finished_df.groupby("net_structure")["Best dice"].mean().to_dict(),
        "success_rate": df["State"].value_counts(normalize=True).to_dict()
    }
    
    print("\n--- ANALYSIS SUMMARY ---")
    print(f"Total experiments: {len(df)}")
    print(f"Success Rate: {summary['success_rate'].get('finished', 0)*100:.1f}%")
    print(f"Best Model: {summary['best_overall']['Name']} ({summary['best_overall']['Best dice']:.4f})")
    print("\nAverage Dice by Architecture:")
    for arch, val in summary['avg_dice_by_arch'].items():
        print(f"  - {arch}: {val:.4f}")
        
    return summary

if __name__ == "__main__":
    analyze_results()

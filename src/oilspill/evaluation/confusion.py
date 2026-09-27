"""Confusion matrix visualization and error analysis utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..data.dataset import CLASS_NAMES


class ConfusionMatrixAnalyzer:
    """Detailed error analysis and visualization on semantic segmentation confusion matrices."""

    def __init__(self, confusion_matrix: np.ndarray, class_names: Optional[Dict[int, str]] = None):
        self.cm = np.array(confusion_matrix, dtype=np.float64)
        self.class_names = [class_names.get(i, f"Class_{i}") if class_names else CLASS_NAMES.get(i, f"Class_{i}")
                            for i in range(self.cm.shape[0])]

    def normalize(self, mode: str = "pred") -> np.ndarray:
        """Normalize confusion matrix.
        
        Args:
            mode: 'pred' (normalize over ground-truth rows, i.e., recall) or 'all' (over total pixels).
            
        Returns:
            Normalized 2D matrix.
        """
        if mode == "pred" or mode == "true":
            row_sums = self.cm.sum(axis=1, keepdims=True)
            norm_cm = np.divide(self.cm, row_sums, out=np.zeros_like(self.cm), where=row_sums != 0)
            return norm_cm
        elif mode == "all":
            total = self.cm.sum()
            return self.cm / total if total > 0 else self.cm
        return self.cm

    def plot(
        self,
        output_path: Optional[Union[str, Path]] = None,
        normalize_mode: str = "true",
        title: str = "Semantic Segmentation Confusion Matrix"
    ) -> plt.Figure:
        """Plot and optionally save a heatmap of the confusion matrix."""
        norm_cm = self.normalize(normalize_mode)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        sns.heatmap(
            norm_cm,
            annot=True,
            fmt=".3f" if normalize_mode else "d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ground Truth Class", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Union[str, Path],
    title: str = "SAR Oil Spill Confusion Matrix"
) -> None:
    """Helper to quickly save confusion matrix plot to disk."""
    analyzer = ConfusionMatrixAnalyzer(cm)
    analyzer.plot(output_path=output_path, title=title)

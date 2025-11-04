# Early stopping callback and evaluation helpers for QCNN training

from math import inf
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from IPython.display import clear_output, display
import pickle

class EarlyStoppingCallback:
    """
    Qiskit ML callback compatible with NeuralNetworkClassifier.
    Tracks objective values and validation accuracy, saves checkpoints,
    and raises an exception to stop training when patience is exceeded.
    
    Extended validation feature:
    - Every epoch: Quick validation on first `quick_val_size` images (for plotting)
    - Every `extended_val_interval` epochs: Extended validation on all validation images
      for more robust best model selection
    """

    def __init__(
        self,
        qnn,
        val_images,
        val_labels,
        output_dir: str,
        saved_model_name: str,
        save_every_x_epoch: int = 50,
        patience: int = 20,
        plot: bool = True,
        quick_val_size: int = 15,
        extended_val_interval: int = 5,
    ) -> None:
        self.qnn = qnn
        self.val_images = val_images
        self.val_labels = val_labels
        self.output_dir = output_dir
        self.saved_model_name = saved_model_name
        self.save_every_x_epoch = save_every_x_epoch
        self.patience = patience
        self.plot = plot
        
        # Extended validation settings
        self.quick_val_size = quick_val_size
        self.extended_val_interval = extended_val_interval
        self.use_extended_validation = len(val_images) > quick_val_size and extended_val_interval > 0

        if self.patience > 0 and self.quick_val_size <= 0:
            raise ValueError("patience > 0 requires quick_val_size > 0 for early stopping based on validation accuracy.")
        
        if self.use_extended_validation:
            print(f"Extended validation enabled:")
            print(f"  - Quick validation: {quick_val_size} images (every epoch)")
            print(f"  - Extended validation: {len(val_images)} images (every {extended_val_interval} epochs)")
        
        # Training state
        self.objective_func_vals: list[float] = []
        self.val_acc_history: list[float] = []  # Quick validation history (for plotting)
        self.extended_val_acc_history: list[float] = []  # Extended validation history
        self.epoch_idx: int = 0
        self.best_objective_val: float = float("inf")
        self.best_val_acc: float = -np.inf  # Quick validation (for plotting)
        self.best_extended_val_acc: float = -np.inf  # Extended validation
        self.best_weights: Optional[np.ndarray] = None
        self.best_extended_val_weights: Optional[np.ndarray] = None
        self.no_improve_count: int = 0

    # --- helpers ---
    def predict_with_weights(self, X, weights) -> np.ndarray:
        X_np = np.asarray(X)
        y_hat = self.qnn.forward(X_np, weights).reshape(-1)
        y_np = y_hat if isinstance(y_hat, np.ndarray) else y_hat.detach().cpu().numpy()
        preds = np.where(y_np >= 0, 1, -1)
        return preds

    def accuracy(self, X, y, weights) -> float:
        preds = self.predict_with_weights(X, weights)
        y_np = np.asarray(y)
        return float((preds == y_np).mean())

    def _save_checkpoint(self, weights, objective_val: float, suffix: str = "") -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        weight_param_names = None
        try:
            weight_param_names = [str(p) for p in self.qnn.weight_params]
        except Exception:
            weight_param_names = None
        
        model_name = self.saved_model_name if not suffix else self.saved_model_name.replace('.pkl', f'{suffix}.pkl')
        save_weights(
            self.output_dir,
            model_name,
            weights,
            weight_param_names,
            objective_val,
            self.epoch_idx,
        )
        if self.epoch_idx % self.save_every_x_epoch == 0 and not suffix:
            save_weights(
                self.output_dir,
                f"epoch_{self.epoch_idx}_" + self.saved_model_name,
                weights,
                weight_param_names,
                objective_val,
                self.epoch_idx,
            )

    # --- callback ---
    def __call__(self, weights, obj_func_eval):
        # Update histories
        self.objective_func_vals.append(obj_func_eval)

        if self.quick_val_size > 0:
            # Quick validation (every epoch, for plotting)
            try:
                quick_val_acc = self.accuracy(
                    self.val_images[:self.quick_val_size], 
                    self.val_labels[:self.quick_val_size], 
                    weights
                )
            except Exception:
                quick_val_acc = float("nan")
            self.val_acc_history.append(quick_val_acc)
        
        # Extended validation (every extended_val_interval epochs, for best model selection)
        current_extended_val_acc = None
        if self.use_extended_validation and (self.epoch_idx + 1) % self.extended_val_interval == 0:
            try:
                current_extended_val_acc = self.accuracy(
                    self.val_images, 
                    self.val_labels, 
                    weights
                )
                self.extended_val_acc_history.append(current_extended_val_acc)
                
                # Track best extended validation model
                if current_extended_val_acc > self.best_extended_val_acc:
                    self.best_extended_val_acc = current_extended_val_acc
                    self.best_extended_val_weights = weights.copy()
                    # Save extended validation best model
                    self._save_checkpoint(
                        self.best_extended_val_weights, 
                        obj_func_eval, 
                        suffix="_best_ext_val"
                    )
            except Exception as e:
                print(f"Warning: Extended validation failed at epoch {self.epoch_idx}: {e}")

        # Plot objective and validation accuracy
        if self.plot:
            # Clear previous output to update a single live plot
            clear_output(wait=True)
            fig = self.plot_histories(show=False)
            if fig is not None:
                display(fig)
                plt.close(fig)

        self.epoch_idx += 1

        # Save by objective improvement or periodic checkpoint
        improved_objective = obj_func_eval < self.best_objective_val
        if improved_objective or self.epoch_idx % self.save_every_x_epoch == 0:
            if improved_objective:
                self.best_objective_val = obj_func_eval
            self.best_weights = weights.copy()
            self._save_checkpoint(self.best_weights, self.best_objective_val)

        if self.patience > 0 and self.quick_val_size > 0:
            # Early stopping on quick validation accuracy
            if not np.isnan(quick_val_acc):
                if quick_val_acc > self.best_val_acc:
                    self.best_val_acc = quick_val_acc
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1

            if self.no_improve_count >= self.patience:
                raise RuntimeError("EARLY_STOPPING_TRIGGERED")
    
    # --- accessors ---
    def get_best(self) -> Tuple[Optional[np.ndarray], float, float]:
        """Get best model by training loss"""
        return self.best_weights, self.best_objective_val, self.best_val_acc
    
    def get_best_extended_val(self) -> Tuple[Optional[np.ndarray], float]:
        """Get best model by extended validation accuracy"""
        return self.best_extended_val_weights, self.best_extended_val_acc

    def get_histories(self) -> Tuple[list[float], list[float]]:
        """Get training loss and quick validation accuracy histories (for plotting)"""
        return self.objective_func_vals, self.val_acc_history
    
    def get_extended_val_history(self) -> list[float]:
        """Get extended validation accuracy history"""
        return self.extended_val_acc_history

    def plot_histories(self, show: bool = True, title_prefix: str = ""):
        """
        Plot objective and validation accuracy histories.
        Returns the matplotlib Figure or None if no data to plot.
        If show=False, the figure is not shown (caller can decide to plt.show()).
        """
        has_obj = self.objective_func_vals is not None and len(self.objective_func_vals) > 0
        has_val = self.val_acc_history is not None and len(self.val_acc_history) > 0
        if not (has_obj or has_val):
            return None

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title(f"{title_prefix}Objective function vs iteration".strip())
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Objective")
        if has_obj:
            ax1.plot(range(len(self.objective_func_vals)), self.objective_func_vals, color='tab:blue')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title(f"{title_prefix}Validation accuracy vs iteration".strip())
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Val acc")
        ax2.set_ylim(0.0, 1.0)
        if has_val:
            ax2.plot(range(len(self.val_acc_history)), self.val_acc_history, color='tab:green')

        fig.tight_layout()
        if show:
            plt.show()
        return fig


def save_weights(output_dir, model_name, weights, weight_param_names, best_obj_val, epoch_idx):
    with open(os.path.join(output_dir, model_name), 'wb') as f:
        pickle.dump({
            'weights': weights,
            'weight_param_names': weight_param_names,
            'objective_value': best_obj_val,
            'epoch' : epoch_idx,
        }, f)


class SPSACallbackWrapper:
    """
    Wrapper to adapt EarlyStoppingCallback for SPSA's callback signature.
    SPSA expects: callback(nfev, parameters, fval, stepsize, accepted)
    This wrapper converts it to the format expected by EarlyStoppingCallback.
    """
    def __init__(self, early_stopping_callback):
        self.early_stopping_callback = early_stopping_callback
    
    def __call__(self, nfev, parameters, fval, stepsize, accepted):
        """SPSA callback signature"""
        # Call the EarlyStoppingCallback with the converted signature
        self.early_stopping_callback(parameters, fval)

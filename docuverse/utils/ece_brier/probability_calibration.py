"""
Probability Calibration to Minimize Expected Calibration Error (ECE)

This module implements multiple calibration methods to rescale probabilities 
to the 0-1 range in a way that minimizes ECE scores.

Methods implemented:
1. Temperature Scaling - Simple and effective for neural networks
2. Platt Scaling - Sigmoid-based calibration
3. Isotonic Regression - Non-parametric flexible calibration
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


class ECECalculator:
    """Calculate Expected Calibration Error (ECE)"""
    
    def __init__(self, n_bins=15):
        """
        Initialize ECE calculator
        
        Args:
            n_bins: Number of bins to use for ECE calculation (default: 15)
        """
        self.n_bins = n_bins
    
    def calculate(self, probs, labels, return_details=False):
        """
        Calculate Expected Calibration Error
        
        Args:
            probs: Predicted probabilities (numpy array)
            labels: True labels (numpy array)
            return_details: If True, return detailed bin information
            
        Returns:
            ECE score (float), or (ECE, bin_details) if return_details=True
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_details = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = np.logical_and(probs > bin_lower, probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_details.append({
                    'range': (bin_lower, bin_upper),
                    'count': in_bin.sum(),
                    'accuracy': accuracy_in_bin,
                    'confidence': avg_confidence_in_bin,
                    'error': np.abs(avg_confidence_in_bin - accuracy_in_bin)
                })
        
        if return_details:
            return ece, bin_details
        return ece


class TemperatureScaling:
    """
    Temperature Scaling calibration method
    
    This method divides the logits (or applies a scaling to probabilities)
    by a learned temperature parameter to improve calibration.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probs, labels):
        """
        Fit temperature parameter to minimize ECE
        
        Args:
            probs: Predicted probabilities (numpy array)
            labels: True labels (numpy array)
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()
        
        # Convert probabilities to logits (inverse sigmoid)
        eps = 1e-7
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        
        # Optimize temperature to minimize negative log likelihood
        def objective(temp):
            scaled_probs = expit(logits / temp)
            # Negative log likelihood
            nll = -np.mean(labels * np.log(scaled_probs + eps) + 
                          (1 - labels) * np.log(1 - scaled_probs + eps))
            return nll
        
        # Find optimal temperature
        result = minimize(objective, x0=1.0, bounds=[(0.01, 1000.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        return self
    
    def transform(self, probs):
        """
        Apply temperature scaling to probabilities
        
        Args:
            probs: Predicted probabilities (numpy array)
            
        Returns:
            Calibrated probabilities (numpy array)
        """
        probs = np.asarray(probs).flatten()
        eps = 1e-7
        probs_clipped = np.clip(probs, eps, 1 - eps)
        
        # Convert to logits, scale, and convert back
        logits = np.log(probs_clipped / (1 - probs_clipped))
        scaled_logits = logits / self.temperature
        calibrated = expit(scaled_logits)
        
        return calibrated
    
    def fit_transform(self, probs, labels):
        """Fit and transform in one step"""
        return self.fit(probs, labels).transform(probs)


class PlattScaling:
    """
    Platt Scaling calibration method
    
    Fits a logistic regression model to map raw predictions to 
    calibrated probabilities.
    """
    
    def __init__(self):
        self.model = LogisticRegression()
        self.A = None
        self.B = None
    
    def fit(self, probs, labels):
        """
        Fit Platt scaling parameters
        
        Args:
            probs: Predicted probabilities (numpy array)
            labels: True labels (numpy array)
        """
        probs = np.asarray(probs).flatten().reshape(-1, 1)
        labels = np.asarray(labels).flatten()
        
        # Fit logistic regression on the probabilities
        self.model.fit(probs, labels)
        
        # Extract parameters
        self.A = self.model.coef_[0][0]
        self.B = self.model.intercept_[0]
        
        return self
    
    def transform(self, probs):
        """
        Apply Platt scaling to probabilities
        
        Args:
            probs: Predicted probabilities (numpy array)
            
        Returns:
            Calibrated probabilities (numpy array)
        """
        probs = np.asarray(probs).flatten().reshape(-1, 1)
        calibrated = self.model.predict_proba(probs)[:, 1]
        return calibrated
    
    def fit_transform(self, probs, labels):
        """Fit and transform in one step"""
        return self.fit(probs, labels).transform(probs)


class IsotonicCalibration:
    """
    Isotonic Regression calibration method
    
    Fits a non-parametric monotonic function to map predictions
    to calibrated probabilities.
    """
    
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, probs, labels):
        """
        Fit isotonic regression
        
        Args:
            probs: Predicted probabilities (numpy array)
            labels: True labels (numpy array)
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()
        
        self.model.fit(probs, labels)
        return self
    
    def transform(self, probs):
        """
        Apply isotonic regression to probabilities
        
        Args:
            probs: Predicted probabilities (numpy array)
            
        Returns:
            Calibrated probabilities (numpy array)
        """
        probs = np.asarray(probs).flatten()
        calibrated = self.model.predict(probs)
        # Ensure output is in [0, 1]
        calibrated = np.clip(calibrated, 0, 1)
        return calibrated
    
    def fit_transform(self, probs, labels):
        """Fit and transform in one step"""
        return self.fit(probs, labels).transform(probs)


class ProbabilityCalibrator:
    """
    Main calibration class that compares different methods
    """
    
    def __init__(self, n_bins=15):
        """
        Initialize calibrator with multiple methods
        
        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.ece_calc = ECECalculator(n_bins=n_bins)
        self.methods = {
            'temperature_scaling': TemperatureScaling(),
            'platt_scaling': PlattScaling(),
            'isotonic': IsotonicCalibration()
        }
        self.best_method = None
        self.results = {}
    
    def calibrate(self, probs, labels, dev_probs=None, dev_labels=None, method='auto'):
        """
        Calibrates the given probability predictions using the specified method or chooses the best
        method automatically using 10-fold cross validation. The method adjusts the predicted 
        probabilities to better match the empirical likelihood of the outcomes.

        Args:
            probs (np.ndarray): Array of predicted probabilities. It should be a 1-dimensional
                array or will be flattened to 1D.
            labels (np.ndarray): Array of true labels corresponding to the predicted probabilities.
                It should be a 1-dimensional array or will be flattened to 1D.
            dev_probs (np.ndarray, optional): Array of development set predicted probabilities.
                Default is None. If None, the `probs` array is used instead.
            dev_labels (np.ndarray, optional): Array of development set true labels corresponding
                to the development set predicted probabilities. Default is None. If None, the
                `labels` array is used instead.
            method (str, optional): Method to use for calibration. If 'auto', all available methods
                will be evaluated and the best one will be selected. If a specific method name is
                provided, only that method will be used. Default is 'auto'.

        Returns:
            dict: A dictionary containing the calibrated probabilities for each method evaluated.
                If 'auto' is specified for the method, it also includes the best-calibrated
                probabilities and cross-validation scores.

        Raises:
            ValueError: If a method is specified that is not recognized or supported.
        """
        probs = np.asarray(probs).flatten()
        labels = np.asarray(labels).flatten()
        if dev_probs is not None:
            dev_probs = np.asarray(dev_probs).flatten()
            dev_labels = np.asarray(dev_labels).flatten()
        else:
            dev_probs = probs
            dev_labels = labels

        # Calculate original ECE
        original_ece = self.ece_calc.calculate(probs, labels)

        if method == 'auto':
            # Try all methods with cross validation
            best_ece = original_ece
            best_calibrated = probs.copy()
            calibrated_dev_probs = {}
            calibrated_test_probs = {}
            calibrated_test_ece = {}
            calibrated_dev_ece = {}
            cv_scores = {}

            kf = KFold(n_splits=10, shuffle=True, random_state=42)

            for method_name, calibrator in self.methods.items():
                try:
                    # Perform cross validation
                    fold_scores = []
                    for train_idx, val_idx in kf.split(dev_probs):
                        # Split data
                        train_probs = dev_probs[train_idx]
                        train_labels = dev_labels[train_idx]
                        val_probs = dev_probs[val_idx]
                        val_labels = dev_labels[val_idx]

                        # Train on fold
                        calibrator.fit(train_probs, train_labels)

                        # Evaluate on validation
                        val_calibrated = calibrator.transform(val_probs)
                        fold_ece = self.ece_calc.calculate(val_calibrated, val_labels)
                        fold_scores.append(fold_ece)

                    # Store cross validation scores
                    cv_scores[method_name] = {
                        'mean': np.mean(fold_scores),
                        'std': np.std(fold_scores),
                        'scores': fold_scores
                    }

                    # Fit on full dev set and transform test set
                    calibrated_dev_probs[method_name] = calibrator.fit_transform(dev_probs, dev_labels)
                    calibrated_dev_ece[method_name] = self.ece_calc.calculate(calibrated_dev_probs[method_name],
                                                                              dev_labels)
                    calibrated_test_probs[method_name] = calibrator.transform(probs)
                    calibrated_test_ece[method_name] = self.ece_calc.calculate(calibrated_test_probs[method_name],
                                                                               labels)

                    # Use mean CV score for method selection
                    ece = cv_scores[method_name]['mean']

                    self.results[method_name] = {
                        'ece': ece,
                        'improvement': original_ece - ece,
                        'cv_scores': cv_scores[method_name]
                    }

                    if ece < best_ece:
                        best_ece = ece
                        best_calibrated = calibrated_test_probs[method_name]
                        self.best_method = method_name

                except Exception as e:
                    print(f"Warning: {method_name} failed with error: {e}")
                    self.results[method_name] = {
                        'ece': np.inf,
                        'improvement': -np.inf,
                        'cv_scores': {'mean': np.inf, 'std': np.inf, 'scores': []}
                    }

            if self.best_method is None:
                self.best_method = 'none (original)'
                best_calibrated = probs

            self.results['original'] = {'ece': original_ece, 'improvement': 0.0}

            return {
                'test': {'probs': calibrated_test_probs, 'ece': calibrated_test_ece},
                'dev': {'probs': calibrated_dev_probs, 'ece': calibrated_dev_ece},
                'cv_scores': cv_scores
            }
        else:
            # Use specified method
            if method not in self.methods:
                raise ValueError(f"Unknown method: {method}. Choose from {list(self.methods.keys())} or 'auto'")

            calibrated_probs = {method: self.methods[method].fit_transform(probs, labels)}
            calibrated_ece = {method: self.ece_calc.calculate(calibrated_probs, labels)}
            
            self.results = {
                'original': {'ece': original_ece, 'improvement': 0.0},
                method: {'ece': calibrated_ece, 'improvement': original_ece - calibrated_ece}
            }
            self.best_method = method
            
            return {'test': {'probs': calibrated_probs, 'ece': calibrated_ece}}
    
    def print_results(self):
        """Print calibration results"""
        if not self.results:
            print("No calibration has been performed yet.")
            return
        
        print("\n" + "="*60)
        print("PROBABILITY CALIBRATION RESULTS")
        print("="*60)
        
        # Sort by ECE (ascending)
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['ece'])
        
        for method_name, result in sorted_results:
            improvement_pct = result['improvement'] / self.results['original']['ece'] * 100 if self.results['original']['ece'] > 0 else 0
            
            marker = " ✓ BEST" if method_name == self.best_method else ""
            
            print(f"\n{method_name.upper()}{marker}")
            print(f"  ECE: {result['ece']:.6f}")
            if method_name != 'original':
                print(f"  Improvement: {result['improvement']:.6f} ({improvement_pct:+.2f}%)")
        
        print("\n" + "="*60)
        print(f"RECOMMENDED METHOD: {self.best_method}")
        print("="*60 + "\n")


def create_reliability_diagram_data(probs, labels, n_bins=10):
    """
    Create data for a reliability diagram (calibration curve)
    
    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
    
    Returns:
        Dictionary with bin data for plotting
    """
    probs = np.asarray(probs).flatten()
    labels = np.asarray(labels).flatten()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs > bin_lower, probs <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(labels[in_bin].mean())
            bin_confidences.append(probs[in_bin].mean())
            bin_counts.append(in_bin.sum())
    
    return {
        'bin_centers': np.array(bin_centers),
        'accuracies': np.array(bin_accuracies),
        'confidences': np.array(bin_confidences),
        'counts': np.array(bin_counts)
    }


# Example usage
if __name__ == "__main__":
    print("Probability Calibration for ECE Minimization")
    print("=" * 60)
    
    # Generate synthetic example data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate poorly calibrated probabilities (overconfident model)
    true_labels = np.random.binomial(1, 0.5, n_samples)
    # Add bias to make predictions overconfident
    raw_probs = true_labels * np.random.beta(8, 2, n_samples) + \
                (1 - true_labels) * np.random.beta(2, 8, n_samples)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)
    
    print(f"\nGenerated {n_samples} samples with overconfident predictions")
    print(f"True positive rate: {true_labels.mean():.3f}")
    print(f"Average predicted probability: {raw_probs.mean():.3f}")
    
    # Create calibrator
    calibrator = ProbabilityCalibrator(n_bins=15)
    
    # Perform calibration (automatically chooses best method)
    print("\nPerforming calibration...")
    calibrated_probs = calibrator.calibrate(raw_probs, true_labels, method='auto')['test']['probs']
    
    # Print results
    calibrator.print_results()
    
    # Show first 10 examples
    print("Example calibrated probabilities (first 10 samples):")
    print("-" * 60)
    print(f"{'Original':<15} {'Calibrated':<15} {'True Label':<15}")
    print("-" * 60)
    for i in range(10):
        print(f"{raw_probs[i]:<15.4f} {calibrated_probs[i]:<15.4f} {true_labels[i]:<15}")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("""
To use this calibrator with your own data:

1. Prepare your data:
   - probs: array of predicted probabilities (0-1 range)
   - labels: array of true binary labels (0 or 1)

2. Create calibrator:
   calibrator = ProbabilityCalibrator(n_bins=15)

3. Calibrate probabilities:
   # Auto-select best method
   calibrated = calibrator.calibrate(probs, labels, method='auto')
   
   # Or specify a method
   calibrated = calibrator.calibrate(probs, labels, method='temperature_scaling')

4. View results:
   calibrator.print_results()

Available methods:
- 'temperature_scaling': Best for neural networks
- 'platt_scaling': Good for SVMs and general use  
- 'isotonic': Flexible, needs more data
- 'auto': Automatically chooses the best method
    """)

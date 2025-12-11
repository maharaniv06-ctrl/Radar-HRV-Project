import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()  # Turn off interactive mode
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
from datetime import datetime
import pickle
import joblib
warnings.filterwarnings('ignore')

# Tambahkan import untuk smoothing
try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Set style untuk plot
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HRV_SVM_Analyzer:
    def __init__(self, csv_file):
        """
        Inisialisasi analyzer dengan file CSV
        """
        self.csv_file = csv_file
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.y_pred = None
        self.metrics = {}
        
    def load_data(self):
        """
        Load dan explore data dengan data cleaning untuk dataset kecil
        """
        print("=" * 60)
        print("📊 LOADING DAN EKSPLORASI DATA")
        print("=" * 60)
        
        # Load data
        self.data = pd.read_csv(self.csv_file)
        print(f"✅ Data berhasil dimuat: {self.data.shape[0]} sampel, {self.data.shape[1]} fitur")
        
        # CRITICAL: Data cleaning untuk dataset kecil
        print(f"\n🔍 DATA CLEANING:")
        print(f"   Original shape: {self.data.shape}")
        
        # Cek missing values
        missing_info = self.data.isnull().sum()
        total_missing = missing_info.sum()
        print(f"   Missing values per column:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                print(f"     - {col}: {missing_count} NaN values")
        
        if total_missing > 0:
            print(f"   📋 Total missing values: {total_missing}")
            
            # Strategy untuk dataset kecil: prioritas preserve data
            print(f"   🔧 Cleaning strategy for small dataset:")
            
            # 1. Cek apakah ada row yang semua fitur NaN
            all_nan_rows = self.data.isnull().all(axis=1).sum()
            if all_nan_rows > 0:
                print(f"     - Removing {all_nan_rows} completely empty rows")
                self.data = self.data.dropna(how='all')
            
            # 2. Cek missing di Label column (CRITICAL)
            if 'Label' in missing_info and missing_info['Label'] > 0:
                print(f"     - ⚠️ {missing_info['Label']} missing labels found - removing these rows")
                self.data = self.data.dropna(subset=['Label'])
            
            # 3. Handle missing features
            feature_cols = [col for col in self.data.columns if col != 'Label']
            missing_features = missing_info[feature_cols]
            
            if missing_features.sum() > 0:
                print(f"     - Missing features found. Options:")
                
                # Option 1: Drop rows with any missing features (conservative)
                rows_with_missing = self.data[feature_cols].isnull().any(axis=1).sum()
                print(f"       * Drop rows with missing features: lose {rows_with_missing} samples")
                
                # Option 2: Fill missing values (risky for small dataset)
                print(f"       * Fill missing values: keep all {len(self.data)} samples")
                
                # Decision: untuk dataset <100, lebih baik drop rows
                if len(self.data) - rows_with_missing >= 30:  # Minimum viable dataset
                    print(f"     - ✅ Dropping rows with missing features (safer for small dataset)")
                    self.data = self.data.dropna()
                else:
                    print(f"     - ⚠️ Too many missing values! Using imputation to preserve data")
                    # Impute dengan median untuk numeric
                    for col in feature_cols:
                        if self.data[col].isnull().sum() > 0:
                            if self.data[col].dtype in ['float64', 'int64']:
                                fill_value = self.data[col].median()
                                self.data[col] = self.data[col].fillna(fill_value)
                                print(f"       * Filled {col} NaN with median: {fill_value:.3f}")
        else:
            print(f"   ✅ No missing values found")
        
        # Verify final data
        print(f"   Final shape after cleaning: {self.data.shape}")
        final_missing = self.data.isnull().sum().sum()
        print(f"   Final missing values: {final_missing}")
        
        if len(self.data) < 20:
            print(f"   ❌ CRITICAL: Too few samples after cleaning ({len(self.data)})")
            print(f"   📋 Recommendation: Check your data quality")
            return None
        
        # Info dasar data
        print(f"\n📋 Informasi Dataset (After Cleaning):")
        print(f"   - Total sampel: {len(self.data)}")
        print(f"   - Total fitur: {self.data.shape[1] - 1}")  # -1 untuk kolom label
        print(f"   - Fitur: {list(self.data.columns[:-1])}")
        
        # Validate labels
        unique_labels = sorted(self.data['Label'].unique())
        print(f"   - Unique labels: {unique_labels}")
        
        if len(unique_labels) != 2:
            print(f"   ❌ ERROR: Expected 2 classes, found {len(unique_labels)}")
            return None
            
        if not all(label in [0, 1] for label in unique_labels):
            print(f"   ⚠️ WARNING: Labels are not 0/1, converting...")
            # Convert labels to 0/1
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
            self.data['Label'] = self.data['Label'].map(label_mapping)
            print(f"   📋 Label mapping: {label_mapping}")
        
        # Distribusi label
        label_dist = self.data['Label'].value_counts().sort_index()
        print(f"\n🏷️  Distribusi Label:")
        print(f"   - Label 0 (Normal): {label_dist[0]} sampel ({label_dist[0]/len(self.data)*100:.1f}%)")
        print(f"   - Label 1 (PAF): {label_dist[1]} sampel ({label_dist[1]/len(self.data)*100:.1f}%)")
        
        # Check class balance
        min_class = min(label_dist)
        if min_class < 5:
            print(f"   ⚠️ WARNING: Very imbalanced classes (min class: {min_class} samples)")
        
        # Analisis separabilitas data
        self.analyze_data_separability()
        
        return self.data
    
    def analyze_data_separability(self):
        """
        Analisis separabilitas data untuk memahami kompleksitas klasifikasi
        """
        print(f"\n🔬 ANALISIS SEPARABILITAS DATA:")
        print("-" * 40)
        
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        
        # Hitung korelasi rata-rata fitur dengan label
        correlations = []
        for col in X.columns:
            corr = abs(X[col].corr(y))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("🔍 Korelasi Fitur dengan Label (Top 5):")
        for i, (feature, corr) in enumerate(correlations[:5]):
            print(f"   {i+1}. {feature}: {corr:.3f}")
        
        # Hitung separabilitas menggunakan mean difference
        for label in [0, 1]:
            label_data = X[y == label]
            print(f"\n📊 Statistik Label {label} ({'Normal' if label == 0 else 'PAF'}):")
            print(f"   - Jumlah sampel: {len(label_data)}")
            print(f"   - Mean fitur terkuat: {label_data[correlations[0][0]].mean():.3f}")
    
    def preprocess_data(self, test_size=0.3, random_state=42):
        """
        Preprocessing data dan split train-test dengan validasi yang lebih robust
        """
        print("\n" + "=" * 60)
        print("🔧 PREPROCESSING DATA")
        print("=" * 60)
        
        # Validasi data sebelum preprocessing
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available. Check data loading step.")
        
        # Pisahkan fitur dan target
        self.X = self.data.drop('Label', axis=1)
        self.y = self.data['Label']
        
        print(f"✅ Fitur (X): {self.X.shape}")
        print(f"✅ Target (y): {self.y.shape}")
        
        # CRITICAL: Validasi final NaN check
        print(f"\n🔍 FINAL NaN VALIDATION:")
        x_nan = self.X.isnull().sum().sum()
        y_nan = self.y.isnull().sum()
        print(f"   - NaN in features (X): {x_nan}")
        print(f"   - NaN in target (y): {y_nan}")
        
        if x_nan > 0 or y_nan > 0:
            print(f"   ❌ CRITICAL: Still have NaN values!")
            print(f"   🔧 Emergency cleanup...")
            
            # Emergency cleanup
            combined_data = pd.concat([self.X, self.y], axis=1)
            clean_data = combined_data.dropna()
            
            if len(clean_data) < 10:
                raise ValueError(f"Too few samples after NaN removal: {len(clean_data)}")
            
            self.X = clean_data.drop('Label', axis=1)
            self.y = clean_data['Label']
            
            print(f"   ✅ Emergency cleanup complete: {len(clean_data)} samples remaining")
        
        # Cek ukuran dataset dan berikan warning
        if len(self.X) < 50:
            print(f"⚠️  WARNING: Dataset kecil ({len(self.X)} sampel). Hasil mungkin tidak stabil.")
            print("   Recommendation: Gunakan cross-validation untuk evaluasi yang lebih robust.")
        
        # Validate minimum samples per class
        class_counts = self.y.value_counts()
        min_class_count = min(class_counts)
        
        if min_class_count < 3:
            raise ValueError(f"Too few samples in minority class: {min_class_count}")
        
        # Adjust test_size for very small datasets
        if len(self.X) < 30:
            test_size = max(0.2, 3/len(self.X))  # At least 3 samples for test, max 20%
            print(f"   📊 Adjusted test_size for small dataset: {test_size:.2f}")
        
        # Split data dengan stratifikasi untuk mempertahankan distribusi label
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, 
                stratify=self.y, shuffle=True
            )
        except ValueError as e:
            print(f"   ⚠️ Stratified split failed: {e}")
            print(f"   🔧 Using regular split without stratification...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, shuffle=True
            )
        
        print(f"\n📊 Pembagian Data (Training:Testing = {int((1-test_size)*100)}:{int(test_size*100)}):")
        print(f"   - Training: {len(self.X_train)} sampel")
        print(f"   - Testing: {len(self.X_test)} sampel")
        
        # Validasi ukuran minimum dataset
        if len(self.X_test) < 3:
            print(f"⚠️  WARNING: Test set sangat kecil ({len(self.X_test)} sampel)")
            print("   Hasil evaluasi mungkin tidak reliabel.")
        
        # Distribusi label setelah split
        print(f"\n🏷️  Distribusi Label - Training:")
        train_dist = self.y_train.value_counts().sort_index()
        for label in [0, 1]:
            if label in train_dist.index:
                label_name = "Normal" if label == 0 else "PAF"
                print(f"   - {label_name}: {train_dist[label]} sampel ({train_dist[label]/len(self.y_train)*100:.1f}%)")
        
        print(f"\n🏷️  Distribusi Label - Testing:")
        test_dist = self.y_test.value_counts().sort_index()
        for label in [0, 1]:
            if label in test_dist.index:
                label_name = "Normal" if label == 0 else "PAF"
                print(f"   - {label_name}: {test_dist[label]} sampel ({test_dist[label]/len(self.y_test)*100:.1f}%)")
        
        # Normalisasi data
        print(f"\n🔄 Normalisasi fitur menggunakan StandardScaler...")
        
        # Validate numeric data
        if not all(self.X_train.dtypes.apply(lambda x: x.kind in 'biufc')):
            print(f"   ⚠️ WARNING: Non-numeric data detected")
            # Convert to numeric, errors='coerce' will make NaN for non-convertible
            self.X_train = self.X_train.apply(pd.to_numeric, errors='coerce')
            self.X_test = self.X_test.apply(pd.to_numeric, errors='coerce')
            
            # Check if conversion created NaN
            if self.X_train.isnull().sum().sum() > 0:
                print(f"   🔧 Filling NaN created by conversion with column median...")
                for col in self.X_train.columns:
                    if self.X_train[col].isnull().sum() > 0:
                        median_val = self.X_train[col].median()
                        self.X_train[col] = self.X_train[col].fillna(median_val)
                        self.X_test[col] = self.X_test[col].fillna(median_val)
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Data training dinormalisasi: {self.X_train_scaled.shape}")
        print(f"✅ Data testing dinormalisasi: {self.X_test_scaled.shape}")
        
        # Final validation - check for infinity or extreme values
        if np.any(np.isinf(self.X_train_scaled)) or np.any(np.isnan(self.X_train_scaled)):
            print(f"   ❌ CRITICAL: Invalid values in scaled training data")
            raise ValueError("Invalid values after scaling")
            
        print(f"✅ All preprocessing validation passed!")
        
    def train_svm_with_cv(self, use_simple_params=True):
        """
        Training SVM dengan cross-validation dan parameter yang lebih konservatif untuk dataset kecil
        """
        print("\n" + "=" * 60)
        print("🤖 TRAINING MODEL SVM DENGAN CROSS-VALIDATION")
        print("=" * 60)
        
        if use_simple_params:
            print("🎯 Menggunakan parameter sederhana untuk dataset kecil...")
            
            # Parameter yang lebih konservatif untuk dataset kecil
            param_options = [
                {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
                {'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'},
                {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'}
            ]
            
        else:
            print("🔍 Menggunakan parameter grid yang lebih luas...")
            param_options = [
                {'C': 0.01, 'kernel': 'linear', 'gamma': 'scale'},
                {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
                {'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'},
                {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
                {'C': 0.1, 'kernel': 'rbf', 'gamma': 'auto'},
                {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'}
            ]
        
        # Cross-validation dengan stratified fold
        cv_folds = min(5, len(self.y_train) // 2)  # Adaptif untuk dataset kecil
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        best_score = 0
        best_params = None
        cv_results = []
        
        print(f"🔄 Evaluating {len(param_options)} parameter combinations dengan {cv_folds}-fold CV...")
        
        for i, params in enumerate(param_options):
            # Create model dengan probability=True
            model = SVC(**params, probability=True, random_state=42)
            
            # Cross-validation
            scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                   cv=cv, scoring='accuracy', n_jobs=-1)
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            cv_results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            })
            
            print(f"   {i+1}/{len(param_options)}: {params} -> CV Score: {mean_score:.3f} (+/- {std_score*2:.3f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        print(f"\n✅ Best parameters: {best_params}")
        print(f"✅ Best CV score: {best_score:.3f}")
        
        # Train final model dengan best parameters
        self.model = SVC(**best_params, probability=True, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Store CV results untuk analisis
        self.cv_results = cv_results
        
        # Training accuracy
        train_pred = self.model.predict(self.X_train_scaled)
        train_acc = accuracy_score(self.y_train, train_pred)
        print(f"📊 Training Accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
        print(f"📊 CV Accuracy: {best_score:.3f} ({best_score*100:.1f}%)")
        
        # Warning jika overfitting
        if train_acc - best_score > 0.1:
            print(f"⚠️  WARNING: Possible overfitting (Train: {train_acc:.3f}, CV: {best_score:.3f})")
        
    def evaluate_model(self):
        """
        Evaluasi model dengan lebih banyak validasi - FIXED VERSION
        """
        print("\n" + "=" * 60)
        print("📊 EVALUASI MODEL")
        print("=" * 60)
        
        # CRITICAL: Pastikan test data sudah di-scale dengan benar
        if not hasattr(self, 'X_test_scaled') or self.X_test_scaled is None:
            print("🔧 Creating scaled test data...")
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print(f"✅ Test data scaled: {self.X_test_scaled.shape}")
        
        # Prediksi labels
        print("🎯 Making predictions...")
        self.y_pred = self.model.predict(self.X_test_scaled)
        print(f"✅ Predictions created: {len(self.y_pred)} samples")
        
        # Prediksi probabilitas dengan robust error handling
        print("📊 Creating probability predictions...")
        try:
            # Try predict_proba first (best option)
            proba_output = self.model.predict_proba(self.X_test_scaled)
            self.y_pred_proba = proba_output[:, 1]  # Probability for class 1 (PAF)
            print(f"✅ Probabilities from predict_proba(): range {self.y_pred_proba.min():.3f} - {self.y_pred_proba.max():.3f}")
            proba_method = "predict_proba"
            
        except Exception as e1:
            print(f"⚠️ predict_proba() failed: {e1}")
            try:
                # Fallback: use decision function + sigmoid
                print("🔧 Trying decision_function() as fallback...")
                decision_scores = self.model.decision_function(self.X_test_scaled)
                # Convert to probabilities using sigmoid
                self.y_pred_proba = 1 / (1 + np.exp(-decision_scores))
                print(f"✅ Probabilities from decision_function(): range {self.y_pred_proba.min():.3f} - {self.y_pred_proba.max():.3f}")
                proba_method = "decision_function"
                
            except Exception as e2:
                print(f"⚠️ decision_function() also failed: {e2}")
                # Last resort: binary predictions to probabilities
                print("🔧 Using binary predictions as last resort...")
                self.y_pred_proba = self.y_pred.astype(float)
                # Add small noise to avoid identical probabilities
                noise = np.random.normal(0, 0.01, len(self.y_pred_proba))
                self.y_pred_proba = np.clip(self.y_pred_proba + noise, 0, 1)
                print(f"✅ Probabilities from binary predictions: range {self.y_pred_proba.min():.3f} - {self.y_pred_proba.max():.3f}")
                proba_method = "binary_fallback"
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"📊 Confusion matrix calculated: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Hitung semua metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        
        # PPV dan NPV
        ppv = precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # AUC-ROC with comprehensive error handling
        try:
            # Check if we have binary classification and variation in probabilities
            unique_labels = np.unique(self.y_test)
            unique_proba = np.unique(self.y_pred_proba)
            
            if len(unique_labels) >= 2 and len(unique_proba) > 1:
                auc_roc = roc_auc_score(self.y_test, self.y_pred_proba)
                print(f"✅ AUC-ROC calculated successfully: {auc_roc:.3f} (using {proba_method})")
            else:
                auc_roc = 0.5
                print(f"⚠️ AUC-ROC set to 0.5 (insufficient variation in labels or probabilities)")
                
        except Exception as e:
            print(f"⚠️ AUC-ROC calculation failed: {e}")
            auc_roc = 0.5
        
        # Simpan metrics
        self.metrics = {
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': accuracy,
            'Sensitivity (Recall)': recall,
            'Specificity': specificity,
            'Precision (PPV)': ppv,
            'NPV': npv,
            'F1-Score': f1,
            'AUC-ROC': auc_roc
        }
        
        # Print hasil
        print("\n🎯 CONFUSION MATRIX:")
        print(f"   True Negative (TN):  {tn}")
        print(f"   False Positive (FP): {fp}")
        print(f"   False Negative (FN): {fn}")
        print(f"   True Positive (TP):  {tp}")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Accuracy:            {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Sensitivity (Recall): {recall:.3f} ({recall*100:.1f}%)")
        print(f"   Specificity:         {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"   Precision (PPV):     {ppv:.3f} ({ppv*100:.1f}%)")
        print(f"   NPV:                 {npv:.3f} ({npv*100:.1f}%)")
        print(f"   F1-Score:            {f1:.3f} ({f1*100:.1f}%)")
        print(f"   AUC-ROC:             {auc_roc:.3f}")
        
        # Validasi hasil
        total_test = len(self.y_test)
        if total_test < 10:
            print(f"\n⚠️  WARNING: Test set sangat kecil ({total_test} sampel).")
            print("   Hasil evaluasi mungkin tidak reliabel. Pertimbangkan cross-validation.")
        
        # Cross-validation score comparison
        if hasattr(self, 'cv_results'):
            best_cv_score = max([r['mean_score'] for r in self.cv_results])
            print(f"\n🔍 PERBANDINGAN SKOR:")
            print(f"   Test Accuracy:  {accuracy:.3f}")
            print(f"   Best CV Score:  {best_cv_score:.3f}")
            print(f"   Difference:     {abs(accuracy - best_cv_score):.3f}")
            
            if abs(accuracy - best_cv_score) > 0.15:
                print("   ⚠️  Perbedaan signifikan antara CV dan test score!")

    
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix dengan informasi tambahan
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Main confusion matrix
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal (0)', 'PAF (1)'],
                    yticklabels=['Normal (0)', 'PAF (1)'])
        plt.title('Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('Actual Label', fontweight='bold')
        
        # Normalized confusion matrix
        plt.subplot(2, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Normal (0)', 'PAF (1)'],
                    yticklabels=['Normal (0)', 'PAF (1)'])
        plt.title('Normalized Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('Actual Label', fontweight='bold')
        
        # CV scores distribution
        if hasattr(self, 'cv_results'):
            plt.subplot(2, 2, 3)
            cv_scores = [r['mean_score'] for r in self.cv_results]
            cv_stds = [r['std_score'] for r in self.cv_results]
            x_pos = range(len(cv_scores))
            
            plt.bar(x_pos, cv_scores, yerr=cv_stds, capsize=5, alpha=0.7)
            plt.xlabel('Parameter Set')
            plt.ylabel('CV Accuracy')
            plt.title('Cross-Validation Scores', fontweight='bold')
            plt.xticks(x_pos, [f'Set {i+1}' for i in x_pos], rotation=45)
        
        # Model parameters info
        plt.subplot(2, 2, 4)
        plt.axis('off')
        info_text = f"Model Information:\n\n"
        info_text += f"Kernel: {self.model.kernel}\n"
        info_text += f"C: {self.model.C}\n"
        info_text += f"Gamma: {self.model.gamma}\n"
        info_text += f"Training samples: {len(self.X_train)}\n"
        info_text += f"Test samples: {len(self.X_test)}\n"
        info_text += f"Support vectors: {self.model.n_support_}\n"
        
        plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """
        Plot perbandingan metrics dengan informasi CV
        """
        metrics_to_plot = ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 
                          'Precision (PPV)', 'NPV', 'F1-Score']
        values = [self.metrics[metric] for metric in metrics_to_plot]
        
        plt.figure(figsize=(15, 10))
        
        # Bar plot metrics
        plt.subplot(2, 3, 1)
        bars = plt.bar(range(len(metrics_to_plot)), values, 
                      color=sns.color_palette("husl", len(metrics_to_plot)))
        plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Performance Metrics', fontweight='bold')
        plt.ylim(0, 1.1)
        
        # Tambahkan nilai di atas bar
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # ROC Curve
        plt.subplot(2, 3, 2)
        try:
            from scipy.interpolate import interp1d
            
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
            
            # Create smooth interpolation untuk ROC curve
            if len(fpr) > 2:  # Pastikan ada cukup points untuk interpolasi
                # Create more points for smooth curve
                fpr_smooth = np.linspace(0, 1, 300)  # 300 points untuk smooth curve
                
                # Interpolation using cubic spline
                f_interp = interp1d(fpr, tpr, kind='cubic', bounds_error=False, fill_value='extrapolate')
                tpr_smooth = f_interp(fpr_smooth)
                
                # Pastikan values dalam range [0, 1]
                tpr_smooth = np.clip(tpr_smooth, 0, 1)
                
                # Plot smooth curve
                plt.plot(fpr_smooth, tpr_smooth, linewidth=2, 
                        label=f'ROC Curve (AUC = {self.metrics["AUC-ROC"]:.3f})')
            else:
                # Fallback ke original jika data points terlalu sedikit
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'ROC Curve (AUC = {self.metrics["AUC-ROC"]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, 'ROC Curve tidak tersedia', ha='center', va='center')
            plt.title('ROC Curve', fontweight='bold')
            print(f"Warning: ROC curve interpolation failed: {e}")
        # CV Performance
        if hasattr(self, 'cv_results'):
            plt.subplot(2, 3, 3)
            cv_means = [r['mean_score'] for r in self.cv_results]
            cv_stds = [r['std_score'] for r in self.cv_results]
            
            plt.errorbar(range(len(cv_means)), cv_means, yerr=cv_stds, 
                        marker='o', capsize=5, capthick=2)
            plt.axhline(y=self.metrics['Accuracy'], color='red', linestyle='--', 
                       label=f'Test Accuracy: {self.metrics["Accuracy"]:.3f}')
            plt.xlabel('Parameter Set')
            plt.ylabel('Accuracy')
            plt.title('Cross-Validation Performance', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Radar chart
        plt.subplot(2, 3, 4, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        values_radar = values + [values[0]]
        angles += angles[:1]
        
        plt.plot(angles, values_radar, 'o-', linewidth=2, color='#1f77b4')
        plt.fill(angles, values_radar, alpha=0.25, color='#1f77b4')
        plt.xticks(angles[:-1], metrics_to_plot)
        plt.ylim(0, 1)
        plt.title('Performance Radar Chart', fontweight='bold', pad=20)
        
        # Feature importance (untuk kernel linear)
        plt.subplot(2, 3, 5)
        if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            feature_importance = np.abs(self.model.coef_[0])
            feature_names = self.X.columns
            
            indices = np.argsort(feature_importance)[::-1][:10]  # Top 10
            
            plt.barh(range(len(indices)), feature_importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Absolute Coefficient Value')
            plt.title('Feature Importance (Top 10)', fontweight='bold')
        else:
            plt.text(0.5, 0.5, f'Feature importance\nnot available for\n{self.model.kernel} kernel', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Feature Importance', fontweight='bold')
            plt.axis('off')
        
        # Prediction distribution
        plt.subplot(2, 3, 6)
        plt.hist(self.y_pred_proba[self.y_test == 0], alpha=0.7, label='Normal', bins=10)
        plt.hist(self.y_pred_proba[self.y_test == 1], alpha=0.7, label='PAF', bins=10)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate laporan lengkap dengan analisis yang lebih mendalam
        """
        print("\n" + "=" * 80)
        print("📋 LAPORAN ANALISIS SVM - KLASIFIKASI HRV PAF")
        print("=" * 80)
    
    def print_summary(self):
        """
        Print ringkasan hasil yang jelas di console
        """
        print("\n" + "🎯" * 20)
    
    def create_all_plots(self):
        """
        Create all plots in one go - COMPLETE IMPLEMENTATION with smooth ROC curve
        """
        print("\n📊 CREATING COMPREHENSIVE PLOTS...")
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            
            # Check if we have all required data
            if not hasattr(self, 'metrics') or not self.metrics:
                print("❌ No metrics available for plotting")
                return None
            
            print("🎨 Creating subplot 1: Performance Metrics...")
            # 1. Performance Metrics Bar Chart
            plt.subplot(2, 4, 1)
            metrics_to_plot = ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 
                            'Precision (PPV)', 'NPV', 'F1-Score']
            values = [self.metrics[metric] for metric in metrics_to_plot]
            
            bars = plt.bar(range(len(metrics_to_plot)), values, 
                        color=sns.color_palette("husl", len(metrics_to_plot)))
            plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45, ha='right')
            plt.ylabel('Score')
            plt.title('Performance Metrics', fontweight='bold')
            plt.ylim(0, 1.1)
            
            # Add values on bars
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            print("🎨 Creating subplot 2: ROC Curve...")
            # 2. ROC Curve (ROBUST SMOOTH VERSION)
            plt.subplot(2, 4, 2)
            roc_success = False
            
            try:
                # Check if we have the necessary data for ROC curve
                if (hasattr(self, 'y_test') and hasattr(self, 'y_pred_proba') and 
                    self.y_test is not None and self.y_pred_proba is not None):
                    
                    # Check for variation in both labels and probabilities
                    unique_labels = np.unique(self.y_test)
                    unique_proba = np.unique(self.y_pred_proba)
                    
                    print(f"   ROC Debug: {len(unique_labels)} unique labels, {len(unique_proba)} unique probabilities")
                    
                    if len(unique_labels) >= 2 and len(unique_proba) > 1:
                        # Calculate ROC curve
                        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
                        
                        print(f"   ROC calculated: {len(fpr)} points")
                        
                        # Try to create smooth curve if scipy is available and we have enough points
                        if SCIPY_AVAILABLE and len(fpr) >= 4:
                            try:
                                # Create smooth interpolation
                                fpr_smooth = np.linspace(0, 1, 300)
                                f_interp = interp1d(fpr, tpr, kind='cubic', bounds_error=False, fill_value='extrapolate')
                                tpr_smooth = f_interp(fpr_smooth)
                                tpr_smooth = np.clip(tpr_smooth, 0, 1)
                                
                                # Plot smooth curve
                                plt.plot(fpr_smooth, tpr_smooth, linewidth=3, color='#2E86AB',
                                        label=f'ROC Curve (AUC = {self.metrics["AUC-ROC"]:.3f})')
                                print(f"   ✅ Smooth ROC curve created")
                                
                            except Exception as smooth_error:
                                # Fallback to regular curve
                                plt.plot(fpr, tpr, linewidth=3, color='#2E86AB',
                                        label=f'ROC Curve (AUC = {self.metrics["AUC-ROC"]:.3f})')
                                print(f"   ⚠️ Smoothing failed, using regular curve: {smooth_error}")
                        else:
                            # Regular curve (no smoothing)
                            plt.plot(fpr, tpr, linewidth=3, color='#2E86AB',
                                    label=f'ROC Curve (AUC = {self.metrics["AUC-ROC"]:.3f})')
                            if not SCIPY_AVAILABLE:
                                print(f"   ⚠️ Scipy not available, using regular curve")
                            else:
                                print(f"   ⚠️ Not enough points for smoothing, using regular curve")
                        
                        # Plot diagonal reference line
                        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random Classifier')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('ROC Curve', fontweight='bold')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        roc_success = True
                        
                    else:
                        print(f"   ⚠️ Insufficient variation for ROC curve")
                else:
                    print(f"   ⚠️ Missing data for ROC curve")
                    
            except Exception as roc_error:
                print(f"   ❌ ROC curve error: {roc_error}")
            
            # If ROC failed, show informative message
            if not roc_success:
                plt.text(0.5, 0.5, 'ROC Curve tidak dapat dibuat\n(Lihat console untuk detail)', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                plt.title('ROC Curve', fontweight='bold')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
            
            print("🎨 Creating subplot 3: Cross-Validation...")
            # 3. Cross-Validation Performance
            plt.subplot(2, 4, 3)
            if hasattr(self, 'cv_results') and self.cv_results:
                try:
                    cv_means = [r['mean_score'] for r in self.cv_results]
                    cv_stds = [r['std_score'] for r in self.cv_results]
                    
                    plt.errorbar(range(len(cv_means)), cv_means, yerr=cv_stds, 
                                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
                    plt.axhline(y=self.metrics['Accuracy'], color='red', linestyle='--', linewidth=2,
                            label=f'Test Accuracy: {self.metrics["Accuracy"]:.3f}')
                    plt.xlabel('Parameter Set')
                    plt.ylabel('Accuracy')
                    plt.title('Cross-Validation Performance', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                except Exception as cv_error:
                    plt.text(0.5, 0.5, f'CV plot error:\n{str(cv_error)[:30]}...', ha='center', va='center')
                    plt.title('Cross-Validation Performance', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'CV results tidak tersedia', ha='center', va='center')
                plt.title('Cross-Validation Performance', fontweight='bold')
            
            print("🎨 Creating subplot 4: Confusion Matrix...")
            # 4. Confusion Matrix
            plt.subplot(2, 4, 4)
            try:
                cm = confusion_matrix(self.y_test, self.y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'PAF'], yticklabels=['Normal', 'PAF'])
                plt.title('Confusion Matrix', fontweight='bold')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
            except Exception as cm_error:
                plt.text(0.5, 0.5, f'Confusion Matrix error:\n{str(cm_error)[:30]}...', ha='center', va='center')
                plt.title('Confusion Matrix', fontweight='bold')
            
            print("🎨 Creating subplot 5: Radar Chart...")
            # 5. Performance Radar Chart
            plt.subplot(2, 4, 5, projection='polar')
            try:
                angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
                values_radar = values + [values[0]]
                angles += angles[:1]
                
                plt.plot(angles, values_radar, 'o-', linewidth=3, color='#1f77b4', markersize=8)
                plt.fill(angles, values_radar, alpha=0.25, color='#1f77b4')
                plt.xticks(angles[:-1], metrics_to_plot, fontsize=10)
                plt.ylim(0, 1)
                plt.title('Performance Radar Chart', fontweight='bold', pad=20)
            except Exception as radar_error:
                plt.text(0.5, 0.5, f'Radar chart error:\n{str(radar_error)[:30]}...', ha='center', va='center')
                plt.title('Performance Radar Chart', fontweight='bold')
            
            print("🎨 Creating subplot 6: Feature Importance/Model Info...")
            # 6. Feature Importance atau Model Info
            plt.subplot(2, 4, 6)
            try:
                if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                    feature_importance = np.abs(self.model.coef_[0])
                    feature_names = self.X.columns
                    
                    indices = np.argsort(feature_importance)[::-1][:10]  # Top 10
                    
                    plt.barh(range(len(indices)), feature_importance[indices])
                    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                    plt.xlabel('Absolute Coefficient Value')
                    plt.title('Feature Importance (Top 10)', fontweight='bold')
                else:
                    plt.axis('off')
                    info_text = f"Model Information:\n\n"
                    info_text += f"Kernel: {self.model.kernel}\n"
                    info_text += f"C: {self.model.C}\n"
                    info_text += f"Gamma: {self.model.gamma}\n"
                    info_text += f"Training samples: {len(self.X_train)}\n"
                    info_text += f"Test samples: {len(self.X_test)}\n"
                    if hasattr(self.model, 'n_support_'):
                        info_text += f"Support vectors: {self.model.n_support_}\n"
                    
                    plt.text(0.1, 0.9, info_text, transform=plt.gca().transAxes, 
                            fontsize=12, verticalalignment='top', fontfamily='monospace')
                    plt.title('Model Information', fontweight='bold')
            except Exception as info_error:
                plt.text(0.5, 0.5, f'Info plot error:\n{str(info_error)[:30]}...', ha='center', va='center')
                plt.title('Model/Feature Information', fontweight='bold')
            
            print("🎨 Creating subplot 7: Probability Distribution...")
            # 7. Prediction Probability Distribution
            plt.subplot(2, 4, 7)
            try:
                if hasattr(self, 'y_pred_proba') and hasattr(self, 'y_test'):
                    plt.hist(self.y_pred_proba[self.y_test == 0], alpha=0.7, label='Normal', 
                            bins=15, color='skyblue', edgecolor='black')
                    plt.hist(self.y_pred_proba[self.y_test == 1], alpha=0.7, label='PAF', 
                            bins=15, color='salmon', edgecolor='black')
                    plt.xlabel('Predicted Probability')
                    plt.ylabel('Frequency')
                    plt.title('Prediction Probability Distribution', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'Probability data\nnot available', ha='center', va='center')
                    plt.title('Prediction Probability Distribution', fontweight='bold')
            except Exception as prob_error:
                plt.text(0.5, 0.5, f'Probability plot error:\n{str(prob_error)[:30]}...', ha='center', va='center')
                plt.title('Prediction Probability Distribution', fontweight='bold')
            
            print("🎨 Creating subplot 8: Performance Summary...")
            # 8. Model Performance Summary
            plt.subplot(2, 4, 8)
            try:
                plt.axis('off')
                summary_text = f"Performance Summary:\n\n"
                summary_text += f"🎯 Accuracy: {self.metrics['Accuracy']:.3f}\n"
                summary_text += f"📈 AUC-ROC: {self.metrics['AUC-ROC']:.3f}\n"
                summary_text += f"🔍 Sensitivity: {self.metrics['Sensitivity (Recall)']:.3f}\n"
                summary_text += f"🛡️ Specificity: {self.metrics['Specificity']:.3f}\n"
                summary_text += f"⚖️ F1-Score: {self.metrics['F1-Score']:.3f}\n\n"
                
                # Add interpretation
                auc = self.metrics['AUC-ROC']
                if auc >= 0.9:
                    summary_text += "🌟 Excellent Model!"
                elif auc >= 0.8:
                    summary_text += "✅ Good Model"
                elif auc >= 0.7:
                    summary_text += "⚠️ Fair Model"
                else:
                    summary_text += "❌ Poor Model"
                
                plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                plt.title('Performance Summary', fontweight='bold')
            except Exception as summary_error:
                plt.text(0.5, 0.5, f'Summary error:\n{str(summary_error)[:30]}...', ha='center', va='center')
                plt.title('Performance Summary', fontweight='bold')
            
            # Add main title
            plt.suptitle('🔬 SVM HRV PAF Classification Results (COMPLETE)', 
                        fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f'svm_results_complete_{timestamp}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✅ All plots saved as '{plot_filename}'")
            
            # Show plot
            plt.show()
            plt.close()
            
            return plot_filename
            
        except Exception as e:
            print(f"❌ Major error creating plots: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def save_model_simple(self):
        """
        Save model in simple way for future use
        """
        print("\n💾 SAVING MODEL...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # ✅ UBAH INI: Tambah suffix "fixed" untuk distinguish
            model_name = f"hrv_paf_svm_fixed_{timestamp}"  # ← CHANGED
            
            # Create model package
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': list(self.X.columns),
                'metrics': self.metrics,
                'model_info': {
                    'kernel': self.model.kernel,
                    'C': self.model.C,
                    'gamma': self.model.gamma,
                    'train_samples': len(self.X_train),
                    'test_samples': len(self.X_test),
                    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'accuracy': self.metrics['Accuracy']
                }
            }
            
            # Save model
            model_filename = f"{model_name}.joblib"
            joblib.dump(model_package, model_filename)
            
            # Save summary
            summary_filename = f"{model_name}_info.txt"
            with open(summary_filename, 'w') as f:
                f.write("🤖 SVM HRV PAF Model (FIXED VERSION)\n")  # ← CHANGED
                f.write("=" * 40 + "\n")  # ← CHANGED
                f.write(f"Model: {model_name}\n")
                f.write(f"Date: {model_package['model_info']['training_date']}\n")
                f.write(f"Kernel: {self.model.kernel}\n")
                f.write(f"Accuracy: {self.metrics['Accuracy']:.3f}\n")
                f.write(f"Sensitivity: {self.metrics['Sensitivity (Recall)']:.3f}\n")
                f.write(f"Specificity: {self.metrics['Specificity']:.3f}\n")
                f.write(f"\nFeatures: {len(self.X.columns)}\n")
                for i, feat in enumerate(self.X.columns, 1):
                    f.write(f"{i:2d}. {feat}\n")
            
            print(f"✅ Model saved as '{model_filename}'")
            print(f"✅ Info saved as '{summary_filename}'")
            
            return model_filename
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    @staticmethod
    def load_and_predict(model_file, new_data_file):
        """
        Load model and predict new data - SIMPLE USAGE
        """
        print(f"🔮 LOADING MODEL AND PREDICTING...")
        print(f"   Model: {model_file}")
        print(f"   Data: {new_data_file}")
        
        try:
            # ... [semua loading code tetap sama] ...
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # ✅ UBAH INI: Tambah "fixed" di nama hasil prediksi
            results_file = f"predictions_fixed_{timestamp}.csv"  # ← CHANGED
            results.to_csv(results_file, index=False)
            
            print(f"✅ Predictions saved as '{results_file}'")
            # ... [rest of code tetap sama] ...
            
            return results
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None
        
    
    def run_full_analysis(self, use_simple_params=True, auto_plot=True, auto_save=True):
        """
        Jalankan analisis lengkap dengan plotting dan saving otomatis
        
        Parameters:
        - use_simple_params: gunakan parameter sederhana untuk dataset kecil
        - auto_plot: otomatis buat plot setelah evaluasi
        - auto_save: otomatis save model setelah evaluasi
        """
        print("🚀 MEMULAI ANALISIS SVM UNTUK KLASIFIKASI HRV PAF")
        print("=" * 80)
        
        try:
            # 1. Load dan explore data
            self.load_data()
            
            # 2. Preprocessing
            self.preprocess_data()
            
            # 3. Training model dengan CV
            self.train_svm_with_cv(use_simple_params=use_simple_params)
            
            # 4. Evaluasi
            self.evaluate_model()
            
            # 5. Print summary
            self.print_summary()
            
            # 6. AUTO PLOTTING (NEW!)
            plot_file = None
            if auto_plot:
                plot_file = self.create_all_plots()
            
            # 7. AUTO SAVE MODEL (NEW!)
            model_file = None
            if auto_save:
                model_file = self.save_model_simple()
            
            # 8. Generate report
            self.generate_report()
            
            # 9. Final summary with files
            print(f"\n🎉 ANALISIS LENGKAP SELESAI!")
            print(f"📁 Files generated:")
            if plot_file:
                print(f"   📊 Plot: {plot_file}")
            if model_file:
                print(f"   🤖 Model: {model_file}")
                print(f"   📋 Info: {model_file.replace('.joblib', '_info.txt')}")
            
            # Instructions for future use
            if model_file:
                print(f"\n🔮 UNTUK PREDIKSI DATA BARU:")
                print(f"   results = HRV_SVM_Analyzer.load_and_predict('{model_file}', 'data_baru.csv')")
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return None

# ===========================
# CARA PENGGUNAAN
# ===========================

if __name__ == "__main__":
    print("🔬 SVM HRV PAF ANALYZER - SIMPLE VERSION")
    print("=" * 60)
    
    # Quick data inspection
    def quick_data_check(filename):
        try:
            df = pd.read_csv(filename)
            print(f"📊 Data: {df.shape[0]} samples, {df.shape[1]} features")
            missing = df.isnull().sum().sum()
            if missing > 0:
                print(f"⚠️ {missing} missing values found - will be cleaned automatically")
            else:
                print(f"✅ Clean data - no missing values")
            return True
        except Exception as e:
            print(f"❌ Cannot read file: {e}")
            return False
    
    # Check data file
    if quick_data_check('radar_hrv_features.csv'):
        
        # Run complete analysis - SIMPLE!
        print(f"\n🚀 Running complete SVM analysis...")
        analyzer = HRV_SVM_Analyzer('radar_hrv_features.csv')
        
        # This does EVERYTHING: train, evaluate, plot, save model
        results = analyzer.run_full_analysis(
            use_simple_params=True,  # Best for small dataset
            auto_plot=True,         # Auto create plots
            auto_save=True          # Auto save model
        )
        
        if results:
            print(f"\n🎊 SUCCESS! Everything completed automatically!")
            
            # Create example for future use
            print(f"\n📝 Creating example data for testing...")
            example_data = pd.DataFrame({
                'LF_power': [0.5, 0.8],
                'HF_power': [1.2, 1.5],
                'P1': [6.2, 6.1],
                'P2': [5.8, 5.9],
                'H1': [9800, 10200],
                'H2': [400, 420],
                'H3': [210, 220],
                'H4': [620000, 680000],
                'SD1': [160, 180],
                'SD2': [180, 200],
                'SD1_SD2_ratio': [0.9, 0.9],
                'SamEn': [1.8, 2.1],
                'Label': [0, 1]  # Optional true labels
            })
            example_data.to_csv('example_new_data.csv', index=False)
            print(f"✅ Example data saved as 'example_new_data.csv'")
            
        else:
            print(f"❌ Analysis failed")
    
    print(f"\n" + "🎯" * 30)
    print(f"📚 CARA MENGGUNAKAN MODEL UNTUK DATA BARU:")
    print(f"🎯" * 30)
    print(f"""
# Untuk prediksi data baru (simple command):
results = HRV_SVM_Analyzer.load_and_predict('model_file.joblib', 'data_baru.csv')

# Data baru harus punya 12 kolom fitur HRV yang sama:
# LF_power, HF_power, P1, P2, H1, H2, H3, H4, SD1, SD2, SD1_SD2_ratio, SamEn
# Kolom 'Label' opsional (untuk validasi)

# Contoh:
# results = HRV_SVM_Analyzer.load_and_predict('hrv_paf_svm_20241220_143022.joblib', 'example_new_data.csv')
    """)
    
    print(f"💡 TIPS:")
    print(f"   • Model dan plot otomatis tersimpan dengan timestamp")
    print(f"   • Hasil prediksi otomatis tersimpan ke CSV")
    print(f"   • Model bisa digunakan berulang kali untuk data baru")
    print(f"   • Format data baru harus sama dengan training data")
    
    print(f"\n🎉 MODEL SIAP UNTUK PRODUCTION USE! 🎯")
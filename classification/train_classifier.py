"""
Étape 3b — Entraînement du classifieur XGBoost
  - Classes : extensif (0) / intensif (1)
  - Features : issues de features.csv (extract_features.py)
  - Validation : spatiale (split column) — train=train, val=val, test=test
  - Sortie : model/classifier.pkl + model/scaler.pkl + model/feature_names.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR      = Path(__file__).parent
FEATURES_CSV  = BASE_DIR / "features.csv"
MODEL_DIR     = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

# Features numériques utilisées pour la classification
FEATURE_COLS = [
    "area_ha",
    "B02_mean", "B03_mean", "B04_mean", "B08_mean", "B11_mean",
    "B02_std",  "B03_std",  "B04_std",  "B08_std",  "B11_std",
    "ndvi_mean", "ndvi_std", "ndvi_p25", "ndvi_p75",
    "ndwi_mean", "ndwi_std",
    "ndre_mean", "ndre_std",
    "glcm_contrast", "glcm_homogeneity",
    "nir_red_ratio",
]


def load_split(df: pd.DataFrame, split_name: str) -> tuple:
    """Retourne (X, y) pour un split donné."""
    sub = df[df["split"] == split_name].copy()
    # Filtrer les classes connues
    sub = sub[sub["label"].isin([0, 1])]
    available = [c for c in FEATURE_COLS if c in sub.columns]
    X = sub[available].values.astype(np.float32)
    y = sub["label"].values.astype(int)
    return X, y, sub["parcel_id"].tolist(), available


def train():
    df = pd.read_csv(FEATURES_CSV)
    print(f"📊 Dataset: {len(df)} parcelles")
    print(df.groupby(["split", "systeme"]).size().unstack(fill_value=0))
    print()

    X_train, y_train, ids_train, feat_cols = load_split(df, "train")
    X_val,   y_val,   ids_val,   _         = load_split(df, "val")
    X_test,  y_test,  ids_test,  _         = load_split(df, "test")

    print(f"Train: {len(X_train)} parcelles")
    print(f"Val  : {len(X_val)} parcelles")
    print(f"Test : {len(X_test)} parcelles")
    print()

    # Imputer les NaN (médiane sur train)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Modèle XGBoost
    # scale_pos_weight pour classes déséquilibrées
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Évaluation
    for name, X, y, ids in [
        ("TRAIN", X_train, y_train, ids_train),
        ("VAL",   X_val,   y_val,   ids_val),
        ("TEST",  X_test,  y_test,  ids_test),
    ]:
        if len(X) == 0:
            continue
        preds = clf.predict(X)
        print(f"── {name} ──")
        print(classification_report(y, preds,
              target_names=["extensif", "intensif"], zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y, preds))
        print()

    # Feature importance
    importances = clf.feature_importances_
    fi = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
    print("Top 10 features:")
    for fname, imp in fi[:10]:
        print(f"  {fname:<25} {imp:.4f}")
    print()

    # Sauvegarde
    joblib.dump(clf,      MODEL_DIR / "classifier.pkl")
    joblib.dump(imputer,  MODEL_DIR / "imputer.pkl")
    joblib.dump(scaler,   MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"✅ Modèle sauvegardé dans {MODEL_DIR}/")
    return clf, imputer, scaler, feat_cols


if __name__ == "__main__":
    train()

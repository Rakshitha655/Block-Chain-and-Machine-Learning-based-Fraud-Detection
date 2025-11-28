# run_simulation.py
import joblib
import pandas as pd
import numpy as np
import os
from pbft_blockchain import PBFTNetwork, make_transaction

# CONFIG - adjust if your filenames differ
MODEL_PATH = "rf_model.joblib"
DATA_PATH = "creditcard.csv"
TARGET_COL = "Class"   # the creditcard dataset uses 'Class' as target

def load_model_and_scaler(path):
    """
    Load whatever was saved at `path`. If a dict was saved (e.g. {'model':..., 'scaler':...})
    extract model and scaler. If a model object alone was saved, return it and scaler=None.
    """
    loaded = joblib.load(path)
    model = None
    scaler = None

    # If saved as a dict
    if isinstance(loaded, dict):
        # common keys used in training scripts
        model = loaded.get("model") or loaded.get("clf") or loaded.get("classifier")
        scaler = loaded.get("scaler") or loaded.get("standardizer")
        # if still None, try to find first sklearn estimator object
        if model is None:
            for v in loaded.values():
                # crude check: has predict attribute
                if hasattr(v, "predict"):
                    model = v
                    break
    else:
        model = loaded

    if model is None:
        raise RuntimeError("Could not find a model inside the joblib file. Inspect the saved object.")

    print(f"[Loader] Loaded model: {type(model)}  scaler: {type(scaler)}")
    return model, scaler

def prepare_features(df):
    """
    Prepare the feature matrix X used by the model.
    For the common 'creditcard.csv' dataset (Kaggle), drop 'Class' and 'Time' optionally.
    """
    # Drop target if present
    X = df.copy()
    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL])

    # Many examples drop 'Time' as it is not scaled similarly
    if 'Time' in X.columns:
        X = X.drop(columns=['Time'])

    feature_cols = X.columns.tolist()
    return X.values, feature_cols

def main(num_samples=5):
    # 1) load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file {MODEL_PATH} not found. Please run training first.")
        return

    model, scaler = load_model_and_scaler(MODEL_PATH)

    # 2) load dataset
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file {DATA_PATH} not found.")
        return

    print(f"[Data] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if TARGET_COL in df.columns:
        print(f"[Data] Using target: {TARGET_COL}")
    else:
        print(f"[Data] Warning: target column {TARGET_COL} not present. Continuing...")

    X_all, feature_cols = prepare_features(df)
    print(f"[Data] Num samples: {len(df)}  Num features: {len(feature_cols)}")

    # 3) pick a small set of rows to simulate (to keep output readable)
    sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # 4) Initialize PBFT Network (same module you ran earlier)
    net = PBFTNetwork(n_nodes=4)  # ensure you used same n_nodes rule when testing earlier

    txs = []
    print()
    for idx, row in sample_df.iterrows():
        # Build feature vector for the model
        # Drop target/time columns same as prepare_features
        features = row.copy()
        if TARGET_COL in features:
            features = features.drop(labels=[TARGET_COL])
        if 'Time' in features:
            features = features.drop(labels=['Time'])

        x = features.values.reshape(1, -1).astype(float)

        # Apply scaler if present
        if scaler is not None:
            x = scaler.transform(x)

        pred = model.predict(x)[0]  # 0 -> normal, 1 -> fraud (depends on training)
        amount = float(row.get('Amount', 0.0))

        # Create on-chain-style transaction record
        sender = f"User{idx+1}"
        receiver = f"Merchant{(idx%5)+1}"
        tx = make_transaction(sender, receiver, amount, features={}, ml_flag=int(pred))
        txs.append(tx)

        print(f"[ML] Row {idx} -> ML predicted isFraud={int(pred)} amount={amount:.2f}")

    # 5) propose a block with selected txs and run consensus
    print()
    print(f"[Simulator] Proposing block with {len(txs)} tx(s)...")
    ok = net.propose_block_from_primary(txs)
    print("Consensus success?", ok)

    # 6) demo: query one tx from node0 storage
    print("\n=== DEMO QUERY on node0 contract storage ===")
    for tx in txs:
        rec = net.nodes[0].contract.query(tx.tx_id)
        print("tx_id:", tx.tx_id[:8], "->", rec)

    # 7) summary: chain lengths
    print("\nSimulation finished. Chain lengths per node:")
    for nid, node in net.nodes.items():
        print(f" Node {nid}: blocks = {len(node.chain)} transactions stored = {len(node.contract.store)}")


if __name__ == "__main__":
    main(num_samples=5)

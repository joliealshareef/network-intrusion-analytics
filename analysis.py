"""
Network Intrusion Analytics - NSL-KDD
Author: Jolie Alshareef

Trains a supervised classifier (Random Forest) and an unsupervised anomaly
detector (Isolation Forest) to separate attack traffic from normal traffic,
then exports every number the website shows to results.json.

Run:  python analysis.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results.json"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Standard NSL-KDD mapping of attack names to the four attack families
FAMILY = {
    "normal": "Normal",
    # Denial of Service
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop",
                          "apache2", "mailbomb", "processtable", "udpstorm"]},
    # Probing / scanning
    **{a: "Probe" for a in ["ipsweep", "nmap", "portsweep", "satan", "mscan",
                            "saint"]},
    # Remote to Local
    **{a: "R2L" for a in ["ftp_write", "guess_passwd", "imap", "multihop", "phf",
                          "spy", "warezclient", "warezmaster", "sendmail", "named",
                          "snmpgetattack", "snmpguess", "xlock", "xsnoop",
                          "worm"]},
    # User to Root
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "perl", "rootkit",
                          "httptunnel", "ps", "sqlattack", "xterm"]},
}
FAMILY_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load(name):
    df = pd.read_csv(DATA / name, names=COLUMNS)
    df["family"] = df["label"].map(FAMILY)
    assert df["family"].notna().all(), df.loc[df["family"].isna(), "label"].unique()
    df["is_attack"] = (df["family"] != "Normal").astype(int)
    return df


def encode(train, test):
    cats = ["protocol_type", "service", "flag"]
    drop = ["label", "difficulty", "family", "is_attack", "num_outbound_cmds"]
    X_tr = pd.get_dummies(train.drop(columns=drop), columns=cats)
    X_te = pd.get_dummies(test.drop(columns=drop), columns=cats)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    return X_tr.astype(float), X_te.astype(float)


def metrics(y, pred, score=None):
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    m["false_positive_rate"] = fp / (fp + tn)
    m["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score)
    return {k: (round(float(v), 4) if not isinstance(v, dict) else v)
            for k, v in m.items()}


def roc_points(y, score, n=60):
    fpr, tpr, _ = roc_curve(y, score)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in idx]


def main():
    t0 = time.time()
    train, test = load("KDDTrain+.txt"), load("KDDTest+.txt")
    X_tr, X_te = encode(train, test)
    y_tr, y_te = train["is_attack"].values, test["is_attack"].values

    train_labels = set(train["label"])
    test["unseen"] = ~test["label"].isin(train_labels)

    # ---------- Dataset overview ----------
    overview = {
        "train_rows": len(train), "test_rows": len(test),
        "features_raw": 41, "features_encoded": X_tr.shape[1],
        "attack_types_train": int(train.loc[train.is_attack == 1, "label"].nunique()),
        "attack_types_test": int(test.loc[test.is_attack == 1, "label"].nunique()),
        "unseen_attack_types": sorted(set(test.loc[test.unseen, "label"])),
        "unseen_rows": int(test["unseen"].sum()),
        "family_counts": {
            split: {f: int((df.family == f).sum()) for f in FAMILY_ORDER}
            for split, df in [("train", train), ("test", test)]
        },
        "protocol_attack_rate": {
            p: round(float(g["is_attack"].mean()), 4)
            for p, g in test.groupby("protocol_type")
        },
    }

    # ---------- Model 1: Logistic Regression baseline ----------
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(scaler.transform(X_tr), y_tr)
    lr_score = lr.predict_proba(scaler.transform(X_te))[:, 1]
    lr_m = metrics(y_te, (lr_score >= 0.5).astype(int), lr_score)

    # ---------- Model 2: Random Forest ----------
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    rf_score = rf.predict_proba(X_te)[:, 1]
    rf_pred = (rf_score >= 0.5).astype(int)
    rf_m = metrics(y_te, rf_pred, rf_score)

    # ---------- Model 3: Isolation Forest (unsupervised, trained on normal only) ----------
    normal_tr = X_tr[y_tr == 0]
    contamination = 0.1
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=SEED, n_jobs=-1)
    iso.fit(scaler.transform(normal_tr))
    iso_score = -iso.score_samples(scaler.transform(X_te))  # higher = more anomalous
    iso_pred = (iso.predict(scaler.transform(X_te)) == -1).astype(int)
    iso_m = metrics(y_te, iso_pred, iso_score)

    # ---------- Per-family detection rate (recall) ----------
    def fam_recall(pred):
        out = {}
        for f in FAMILY_ORDER[1:]:
            mask = (test.family == f).values
            out[f] = round(float(pred[mask].mean()), 4)
        out["Normal (false alarm rate)"] = round(float(pred[(test.family == "Normal").values].mean()), 4)
        return out

    def seen_vs_unseen(pred):
        att = test.is_attack.values == 1
        return {
            "seen": round(float(pred[att & ~test.unseen.values].mean()), 4),
            "unseen": round(float(pred[att & test.unseen.values].mean()), 4),
        }

    # ---------- Model 4: Layered (alert if either model flags) ----------
    hyb_pred = np.maximum(rf_pred, iso_pred)
    hyb_m = metrics(y_te, hyb_pred)

    # ---------- Feature importance ----------
    imp = pd.Series(rf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    top_features = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.head(12).items()]

    # ---------- Per-attack-type recall for the RF (attacks with >= 50 test rows) ----------
    per_attack = []
    for lab, g in test[test.is_attack == 1].groupby("label"):
        if len(g) >= 50:
            per_attack.append({
                "attack": lab, "family": FAMILY[lab], "rows": len(g),
                "unseen": bool(g["unseen"].iloc[0]),
                "rf_recall": round(float(rf_pred[g.index].mean()), 4),
                "iso_recall": round(float(iso_pred[g.index].mean()), 4),
                "hyb_recall": round(float(hyb_pred[g.index].mean()), 4),
            })
    per_attack.sort(key=lambda r: -r["rows"])

    # ---------- Sample connections for the interactive inspector ----------
    rng = np.random.default_rng(SEED)
    picks = []
    for fam in FAMILY_ORDER:
        idx = test.index[test.family == fam].to_numpy()
        picks += list(rng.choice(idx, size=min(6, len(idx)), replace=False))
    unseen_idx = test.index[test.unseen].to_numpy()
    picks += list(rng.choice(unseen_idx, size=6, replace=False))
    samples = []
    for i in dict.fromkeys(picks):
        r = test.loc[i]
        samples.append({
            "id": int(i), "label": r.label, "family": r.family, "unseen": bool(r.unseen),
            "protocol": r.protocol_type, "service": r.service, "flag": r.flag,
            "duration": int(r.duration), "src_bytes": int(r.src_bytes),
            "dst_bytes": int(r.dst_bytes), "count": int(r["count"]),
            "serror_rate": float(r.serror_rate), "same_srv_rate": float(r.same_srv_rate),
            "dst_host_srv_count": int(r.dst_host_srv_count),
            "rf_prob": round(float(rf_score[i]), 4),
            "iso_score": round(float(iso_score[i]), 4),
            "iso_flag": int(iso_pred[i]),
        })

    results = {
        "dataset": "NSL-KDD (KDDTrain+ / KDDTest+)",
        "overview": overview,
        "models": {
            "Logistic Regression": {**lr_m, "type": "Supervised baseline",
                                    "roc": roc_points(y_te, lr_score)},
            "Random Forest": {**rf_m, "type": "Supervised",
                              "roc": roc_points(y_te, rf_score),
                              "family_recall": fam_recall(rf_pred),
                              "seen_unseen": seen_vs_unseen(rf_pred)},
            "Isolation Forest": {**iso_m, "type": "Unsupervised anomaly detection",
                                 "roc": roc_points(y_te, iso_score),
                                 "family_recall": fam_recall(iso_pred),
                                 "seen_unseen": seen_vs_unseen(iso_pred),
                                 "contamination": contamination},
            "Layered (RF + IF)": {**hyb_m, "type": "Alert if either model flags",
                                  "family_recall": fam_recall(hyb_pred),
                                  "seen_unseen": seen_vs_unseen(hyb_pred)},
        },
        "top_features": top_features,
        "per_attack": per_attack,
        "samples": samples,
        "runtime_seconds": round(time.time() - t0, 1),
        "sklearn_version": __import__("sklearn").__version__,
    }
    OUT.write_text(json.dumps(results, indent=1))

    # One character per test connection for the website mosaic.
    # value = family_index*8 + rf*4 + iso*2 + unseen, offset into printable ASCII.
    fam_idx = test["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}).values
    code = fam_idx * 8 + rf_pred * 4 + iso_pred * 2 + test["unseen"].values.astype(int)
    order = np.lexsort((-(rf_pred | iso_pred), fam_idx))  # group by family, caught first
    (Path(__file__).parent / "mosaic.txt").write_text("".join(chr(48 + int(c)) for c in code[order]))
    print(json.dumps({k: v for k, v in results.items() if k not in ("samples", "per_attack")},
                     indent=1, default=str)[:6000])
    print("\nper_attack:", json.dumps(per_attack, indent=0)[:3000])


if __name__ == "__main__":
    main()



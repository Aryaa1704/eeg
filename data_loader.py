import mne
import numpy as np
import scipy.io as sio
from pathlib import Path
from config import SFREQ, EVENT_ID, ARTIFACT_CODE, CLASS_NAMES


def load_train(gdf_path):
    raw = _load_raw(gdf_path)
    events = _read_events(raw)
    artifact_positions = set(events[events[:, 2] == ARTIFACT_CODE, 0])
    mi_events = events[np.isin(events[:, 2], list(EVENT_ID.keys()))]
    audit = _make_audit(gdf_path, "TRAIN", raw, mi_events, artifact_positions)
    _print_audit(audit)
    return {"raw": raw, "events": mi_events,
            "artifact_pos": artifact_positions, "audit": audit}


def load_eval(gdf_path, mat_path):
    raw = _load_raw(gdf_path)
    events = _read_events(raw)
    artifact_positions = set(events[events[:, 2] == ARTIFACT_CODE, 0])
    trial_events = _get_trial_positions(events)
    true_labels  = _load_mu_mat(mat_path)

    assert len(trial_events) == len(true_labels), (
        f"Mismatch: GDF {len(trial_events)} trials, MU.mat {len(true_labels)} labels"
    )

    code_map = {1: 769, 2: 770, 3: 771, 4: 772}
    mi_events = np.column_stack([
        trial_events[:, 0],
        np.zeros(len(trial_events), dtype=int),
        np.array([code_map[int(l)] for l in true_labels])
    ])

    audit = _make_audit(gdf_path, "EVAL", raw, mi_events, artifact_positions)
    _print_audit(audit)
    return {"raw": raw, "events": mi_events,
            "artifact_pos": artifact_positions, "audit": audit}


def _load_raw(gdf_path):
    gdf_path = Path(gdf_path)
    assert gdf_path.exists(), f"Not found: {gdf_path}"
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    assert abs(raw.info["sfreq"] - SFREQ) < 1, f"Bad sfreq: {raw.info['sfreq']}"
    eeg_chs = [c for c in raw.ch_names if not c.upper().startswith("EOG")]
    raw.pick(eeg_chs)
    return raw


def _read_events(raw):
    events, mapping = mne.events_from_annotations(raw, verbose=False)
    original_codes = {}
    for description, generated_code in mapping.items():
        try:
            original_codes[generated_code] = int(description)
        except ValueError:
            continue

    if not original_codes:
        raise ValueError("No numeric event markers found in GDF annotations")

    decoded = events.copy()
    decoded[:, 2] = np.array([
        original_codes.get(code, code) for code in events[:, 2]
    ], dtype=decoded.dtype)
    return decoded


def _get_trial_positions(events):
    for code in [768, 783, 769, 770, 771, 772]:
        sub = events[events[:, 2] == code]
        if len(sub) == 288:
            return sub
        if len(sub) > 0:
            return sub
    raise ValueError("Cannot find 288 trial markers in E.gdf")


def _load_mu_mat(mat_path):
    mat_path = Path(mat_path)
    assert mat_path.exists(), f"Not found: {mat_path}"
    mat = sio.loadmat(str(mat_path))
    for key in ["classlabel", "classLabel", "y", "labels", "label"]:
        if key in mat:
            labels = mat[key].flatten().astype(int)
            assert set(labels).issubset({1,2,3,4}), f"Bad labels: {set(labels)}"
            print(f"[MU.MAT] {mat_path.name} | key='{key}' | n={len(labels)} | unique={sorted(set(labels))}")
            return labels
    available = [k for k in mat.keys() if not k.startswith("_")]
    raise KeyError(f"No classlabel in {mat_path.name}. Keys: {available}")


def _make_audit(gdf_path, session, raw, mi_events, artifact_positions):
    art = sum(1 for ev in mi_events if any(abs(ev[0]-ap)<=10 for ap in artifact_positions))
    cc  = {lid: int(np.sum(mi_events[:,2]==code)) for code, lid in EVENT_ID.items()}
    return {"file": Path(gdf_path).name, "session": session,
            "total_trials": len(mi_events), "artifact_trials": art,
            "clean_trials": len(mi_events)-art, "class_counts": cc,
            "n_channels": len(raw.ch_names), "sfreq": raw.info["sfreq"]}


def _print_audit(a):
    print(f"\n[AUDIT] {a['file']} ({a['session']})")
    print(f"  Channels:{a['n_channels']}  sfreq:{a['sfreq']}Hz")
    print(f"  Trials: {a['total_trials']} total | {a['artifact_trials']} artifacts | {a['clean_trials']} clean")
    for lid, cnt in a["class_counts"].items():
        print(f"    Class {lid} ({CLASS_NAMES[lid]}): {cnt}")

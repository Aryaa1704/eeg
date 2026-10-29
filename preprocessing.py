import numpy as np
import mne
from config import BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, EVENT_ID


def bandpass_filter(raw):
    r = raw.copy()
    r.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
             method="iir", iir_params=dict(order=FILTER_ORDER, ftype="butter"),
             verbose=False)
    print(f"[FILTER] {BANDPASS_LOW}-{BANDPASS_HIGH} Hz Butterworth order {FILTER_ORDER}")
    return r


def extract_epochs(raw_f, events, artifact_pos, tmin, tmax):
    sfreq = raw_f.info["sfreq"]
    data, _ = raw_f[:, :]
    n_samp = int((tmax - tmin) * sfreq)
    X, y = [], []

    for ev in events:
        cue, code = ev[0], ev[2]
        label = EVENT_ID.get(code)
        if label is None:
            continue
        if any(abs(cue - ap) <= 10 for ap in artifact_pos):
            continue
        s = int(cue + tmin * sfreq)
        e = s + n_samp
        if s < 0 or e > data.shape[1]:
            continue
        X.append(data[:, s:e])
        y.append(label)

    X, y = np.array(X), np.array(y)
    if len(y):
        print(f"[EPOCHS] {tmin}-{tmax}s -> {len(y)} epochs | "
              + str({int(c): int(np.sum(y==c)) for c in np.unique(y)}))
    return X, y

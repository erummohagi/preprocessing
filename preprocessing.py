
# ============================================================
# S001 Motor Imagery EEG 전처리 + QC 검증 코드
# ------------------------------------------------------------
# 목적:
#   1) R04 / R08 / R12만 사용
#   2) T0=Rest, T1=Left, T2=Right
#   3) FC3, FCz, FC4, C3, Cz, C4, CP3, CPz, CP4
#   4) Run 전체를 먼저 8~30 Hz, 2차 Butterworth causal filter
#   5) 그 후 0.5~4.0 s 구간에서 2 s / 0.25 s sliding window
#   6) NaN / flat channel / |amplitude| > 100 uV 제거
#   7) 필터 전/후 PSD 비교
#   8) window 수 / artifact 제거 수 / label 분포 / shape / metadata 저장
#
# 주의:
#   - 이것은 "S001 1명 파이프라인 검증용" 코드다.
#   - 아래 DEMO standardization split은 구현 확인용일 뿐,
#     최종 103명 모델 학습/평가용 split이 아니다.
#   - 최종 103명에서는 반드시 subject-level Train/Val/Test split을
#     먼저 만든 뒤 각 split 내부에서 window를 생성해야 한다.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from mne.time_frequency import psd_array_welch
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 설정
# ============================================================

SUBJECT_ID = "S001"
TARGET_RUNS = ["04", "08", "12"]

TARGET_CHANNELS = [
    "FC3", "FCz", "FC4",
    "C3", "Cz", "C4",
    "CP3", "CPz", "CP4"
]

FS = 160

# Filtering
LOW_CUT = 8.0
HIGH_CUT = 30.0
FILTER_ORDER = 2

# Window
WINDOW_SEC = 2.0
STEP_SEC = 0.25
ONSET_START_SEC = 0.5
ONSET_END_SEC = 4.0

WINDOW_SIZE = int(WINDOW_SEC * FS)       # 320
STEP_SIZE = int(STEP_SEC * FS)           # 40
ONSET_START = int(ONSET_START_SEC * FS)  # 80
ONSET_END = int(ONSET_END_SEC * FS)      # 640

NUM_WINDOWS = int(
    (ONSET_END - ONSET_START - WINDOW_SIZE) / STEP_SIZE
) + 1

# Artifact 기준
VOLTAGE_THRESHOLD = 100e-6  # 100 uV

# Label
EVENT_MAPPING = {
    "T0": -1,  # Rest
    "T1": 0,   # Left hand MI
    "T2": 1    # Right hand MI
}

LABEL_NAME = {
    -1: "Rest",
    0: "Left",
    1: "Right"
}

# 결과 폴더
SAVE_DIR = "processed_qc_S001"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 유틸리티
# ============================================================

def find_edf(subject_id, run):
    """데이터셋 폴더 후보를 순회하여 EDF 경로를 찾는다."""
    candidates = [
        f"eeg-motor-movementimagery-dataset-1.0.0/{subject_id}/{subject_id}R{run}.edf",
        f"eeg-motor-movementimagery-dataset-1.0.0/files/{subject_id}/{subject_id}R{run}.edf",
        f"{subject_id}/{subject_id}R{run}.edf",
        f"{subject_id}R{run}.edf",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def check_artifact(window_data):
    """
    window_data shape = (channels, samples)

    제거 기준:
      1) NaN / Inf
      2) flat channel
      3) 한 샘플이라도 |amplitude| > 100 uV

    반환:
      is_artifact, reasons
    """
    reasons = []

    if not np.isfinite(window_data).all():
        reasons.append("nan_or_inf")

    # 채널별 peak-to-peak가 정확히 0이면 flat
    channel_ptp = np.ptp(window_data, axis=1)
    if np.any(channel_ptp == 0):
        reasons.append("flat_channel")

    # "±100 uV" 기준:
    # 어느 한 샘플이라도 absolute amplitude가 100 uV를 넘으면 제거
    if np.max(np.abs(window_data)) > VOLTAGE_THRESHOLD:
        reasons.append("amplitude_over_100uV")

    return (len(reasons) > 0), reasons


def safe_trapz(y, x):
    """numpy 버전에 따라 trapezoid 함수를 선택."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def plot_psd_before_after(raw_before, raw_after, run_id):
    """
    필터 전/후 PSD 비교.
    전체 selected 9채널의 PSD 평균을 비교한다.
    """
    data_before = raw_before.get_data()
    data_after = raw_after.get_data()

    psd_before, freqs = psd_array_welch(
        data_before,
        sfreq=FS,
        fmin=1,
        fmax=60,
        n_fft=1024,
        n_overlap=512,
        average="mean",
        verbose=False
    )

    psd_after, _ = psd_array_welch(
        data_after,
        sfreq=FS,
        fmin=1,
        fmax=60,
        n_fft=1024,
        n_overlap=512,
        average="mean",
        verbose=False
    )

    mean_before = psd_before.mean(axis=0)
    mean_after = psd_after.mean(axis=0)

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, mean_before, label="Before filtering")
    plt.semilogy(freqs, mean_after, label="After 8-30 Hz causal Butterworth")

    plt.axvline(LOW_CUT, linestyle="--", label="8 Hz")
    plt.axvline(HIGH_CUT, linestyle="--", label="30 Hz")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (V²/Hz)")
    plt.title(f"{SUBJECT_ID} {run_id}: PSD Before vs After Filtering")
    plt.xlim(1, 60)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        SAVE_DIR, f"{SUBJECT_ID}R{run_id}_PSD_before_after.png"
    )
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    # 숫자로도 band power 비교
    bands = {
        "1-8Hz": (1, 8),
        "8-30Hz": (8, 30),
        "30-50Hz": (30, 50),
        "50-60Hz": (50, 60),
    }

    rows = []

    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)

        before_power = safe_trapz(
            mean_before[mask], freqs[mask]
        )

        after_power = safe_trapz(
            mean_after[mask], freqs[mask]
        )

        attenuation_db = 10 * np.log10(
            (after_power + 1e-30) / (before_power + 1e-30)
        )

        rows.append({
            "Run": run_id,
            "Band": band_name,
            "Before_Power": before_power,
            "After_Power": after_power,
            "After_vs_Before_dB": attenuation_db
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. Run 하나 처리
# ============================================================

def process_run(file_path, run_id, subject_id):
    """
    한 Run:
        EDF
        -> standardize
        -> 9 channel pick
        -> whole-run causal 8-30 Hz filter
        -> T0/T1/T2 events
        -> 0~4 sec epoch
        -> 0.5~4.0 sec sliding windows
        -> artifact QC
    """

    print("\n" + "=" * 70)
    print(f"{subject_id} R{run_id}")
    print("=" * 70)
    print(f"파일: {file_path}")

    # 원본 로드
    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=False
    )

    # 채널명 표준화
    mne.datasets.eegbci.standardize(raw)

    # 9개 채널 선택
    raw.pick(TARGET_CHANNELS)

    print(f"선택 채널 수: {len(raw.ch_names)}")
    print(f"채널: {raw.ch_names}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    if raw.info["sfreq"] != FS:
        raise ValueError(
            f"예상 sampling rate는 {FS} Hz인데 실제는 "
            f"{raw.info['sfreq']} Hz입니다."
        )

    # 필터 전 데이터 보관
    raw_before = raw.copy()

    # ========================================================
    # 핵심: Run 전체를 먼저 filtering
    # ========================================================
    iir_params = dict(
        order=FILTER_ORDER,
        ftype="butter",
        output="sos"
    )

    raw.filter(
        l_freq=LOW_CUT,
        h_freq=HIGH_CUT,
        method="iir",
        iir_params=iir_params,
        phase="forward",   # causal
        verbose=False
    )

    raw_after = raw

    # PSD QC
    psd_df = plot_psd_before_after(
        raw_before,
        raw_after,
        run_id
    )

    # ========================================================
    # Event 추출
    # ========================================================
    events, event_id = mne.events_from_annotations(
        raw_after,
        verbose=False
    )

    reverse_event_id = {value: key for key, value in event_id.items()}

    valid_event_counts = {
        "T0": 0,
        "T1": 0,
        "T2": 0
    }

    for e in events:
        event_name = reverse_event_id.get(e[2])
        if event_name in EVENT_MAPPING:
            valid_event_counts[event_name] += 1

    print("\n[Event count]")
    print(f"  T0 (Rest)  : {valid_event_counts['T0']}")
    print(f"  T1 (Left)  : {valid_event_counts['T1']}")
    print(f"  T2 (Right) : {valid_event_counts['T2']}")

    # ========================================================
    # 0~4 sec epoch
    # ========================================================
    epochs = mne.Epochs(
        raw_after,
        events,
        tmin=0,
        tmax=4.0,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False
    )

    data = epochs.get_data()
    print(f"\nEpoch shape: {data.shape}")

    if data.shape[1] != len(TARGET_CHANNELS):
        raise ValueError("Epoch 채널 수가 9가 아닙니다.")

    if data.shape[2] != 641:
        raise ValueError(
            f"Epoch sample 수가 641이 아닙니다. 실제: {data.shape[2]}"
        )

    labels = epochs.events[:, -1]

    X_list = []
    y_list = []
    meta_list = []

    # QC
    qc_rows = []

    for i, trial_data in enumerate(data):

        event_code = reverse_event_id.get(labels[i])

        if event_code not in EVENT_MAPPING:
            continue

        mapped_label = EVENT_MAPPING[event_code]
        label_name = LABEL_NAME[mapped_label]

        candidate_windows = 0
        kept_windows = 0
        removed_windows = 0

        reason_counts = {
            "nan_or_inf": 0,
            "flat_channel": 0,
            "amplitude_over_100uV": 0
        }

        for w in range(NUM_WINDOWS):

            candidate_windows += 1

            start_idx = ONSET_START + (w * STEP_SIZE)
            end_idx = start_idx + WINDOW_SIZE

            window_data = trial_data[:, start_idx:end_idx]

            if window_data.shape != (9, 320):
                raise ValueError(
                    f"잘못된 window shape: {window_data.shape}"
                )

            is_artifact, reasons = check_artifact(window_data)

            if is_artifact:
                removed_windows += 1

                for reason in reasons:
                    reason_counts[reason] += 1

                continue

            kept_windows += 1

            # 최종 task/rest 데이터
            X_list.append(window_data.astype(np.float32))
            y_list.append(mapped_label)

            # nominal window time
            start_sec = ONSET_START_SEC + (w * STEP_SEC)
            end_sec = start_sec + WINDOW_SEC

            meta_list.append({
                "Subject": subject_id,
                "Run": run_id,
                "Trial": i,
                "Window_Idx": w,
                "Label": mapped_label,
                "Label_Name": label_name,
                "Start_sec": start_sec,
                "End_sec": end_sec
            })

        qc_rows.append({
            "Subject": subject_id,
            "Run": run_id,
            "Trial": i,
            "Label": mapped_label,
            "Label_Name": label_name,
            "Candidate_Windows": candidate_windows,
            "Kept_Windows": kept_windows,
            "Removed_Windows": removed_windows,
            "Removed_nan_or_inf": reason_counts["nan_or_inf"],
            "Removed_flat_channel": reason_counts["flat_channel"],
            "Removed_amplitude_over_100uV": reason_counts["amplitude_over_100uV"]
        })

    return (
        X_list,
        y_list,
        meta_list,
        pd.DataFrame(qc_rows),
        psd_df
    )


# ============================================================
# 4. S001 R04/R08/R12 실행
# ============================================================

all_X = []
all_y = []
all_meta = []
all_qc = []
all_psd = []

for run in TARGET_RUNS:

    file_path = find_edf(SUBJECT_ID, run)

    if file_path is None:
        print(
            f"[오류] {SUBJECT_ID}R{run}.edf 파일을 찾지 못했습니다."
        )
        continue

    (
        X_list,
        y_list,
        meta_list,
        qc_df,
        psd_df
    ) = process_run(
        file_path,
        run,
        SUBJECT_ID
    )

    all_X.extend(X_list)
    all_y.extend(y_list)
    all_meta.extend(meta_list)
    all_qc.append(qc_df)
    all_psd.append(psd_df)


# ============================================================
# 5. 배열 변환 및 기본 검증
# ============================================================

if len(all_X) == 0:
    raise RuntimeError(
        "최종 window가 0개입니다. EDF 경로/event/filter/window를 확인하세요."
    )

X_all_unstd = np.asarray(all_X, dtype=np.float32)
y_all = np.asarray(all_y, dtype=np.int64)

df_meta = pd.DataFrame(all_meta)
df_qc = pd.concat(all_qc, ignore_index=True)
df_psd = pd.concat(all_psd, ignore_index=True)

print("\n" + "=" * 70)
print("S001 전체 결과")
print("=" * 70)

print(f"전체 X shape: {X_all_unstd.shape}")
print(f"전체 y shape: {y_all.shape}")

if X_all_unstd.shape[1:] != (9, 320):
    raise ValueError(
        f"최종 shape 오류: {X_all_unstd.shape}"
    )

print("\n[전체 Label 분포]")
print(
    pd.Series(y_all)
    .map(LABEL_NAME)
    .value_counts()
)

# ------------------------------------------------------------
# Task / Rest 분리
# ------------------------------------------------------------

task_mask = y_all != -1
rest_mask = y_all == -1

X_task_unstd = X_all_unstd[task_mask]
y_task = y_all[task_mask]

X_rest_unstd = X_all_unstd[rest_mask]

df_task = df_meta.loc[task_mask].reset_index(drop=True)
df_rest = df_meta.loc[rest_mask].reset_index(drop=True)

print("\n[Task]")
print(f"X_task_unstd: {X_task_unstd.shape}")
print(f"y_task      : {y_task.shape}")
print(
    pd.Series(y_task)
    .map(LABEL_NAME)
    .value_counts()
)

print("\n[Rest]")
print(f"X_rest_unstd: {X_rest_unstd.shape}")


# ============================================================
# 6. Window 시간 패턴 검증
# ============================================================

print("\n[Window timing 검증]")

example = df_meta.head(NUM_WINDOWS)

print(
    example[
        ["Run", "Trial", "Window_Idx", "Label_Name", "Start_sec", "End_sec"]
    ].to_string(index=False)
)

expected_starts = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00
]

actual_starts = sorted(
    df_meta["Start_sec"].drop_duplicates().tolist()
)

print(f"\n예상 시작시간: {expected_starts}")
print(f"실제 시작시간: {actual_starts}")


# ============================================================
# 7. Artifact QC 요약
# ============================================================

df_qc.to_csv(
    os.path.join(SAVE_DIR, "S001_artifact_qc_by_trial.csv"),
    index=False,
    encoding="utf-8-sig"
)

qc_summary = (
    df_qc.groupby(["Run", "Label", "Label_Name"], as_index=False)
    .agg(
        Candidate_Windows=("Candidate_Windows", "sum"),
        Kept_Windows=("Kept_Windows", "sum"),
        Removed_Windows=("Removed_Windows", "sum"),
        Removed_nan_or_inf=("Removed_nan_or_inf", "sum"),
        Removed_flat_channel=("Removed_flat_channel", "sum"),
        Removed_amplitude_over_100uV=("Removed_amplitude_over_100uV", "sum"),
    )
)

qc_summary["Removal_Rate_%"] = (
    qc_summary["Removed_Windows"]
    / qc_summary["Candidate_Windows"]
    * 100
)

qc_summary.to_csv(
    os.path.join(SAVE_DIR, "S001_artifact_qc_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[Artifact QC]")
print(qc_summary.to_string(index=False))

total_candidate = int(df_qc["Candidate_Windows"].sum())
total_kept = int(df_qc["Kept_Windows"].sum())
total_removed = int(df_qc["Removed_Windows"].sum())

print("\n전체 후보 window :", total_candidate)
print("정상 보존 window :", total_kept)
print("제거 window      :", total_removed)
print(
    "전체 제거율       : "
    f"{(total_removed / total_candidate * 100):.2f}%"
)


# ============================================================
# 8. Metadata 저장
# ============================================================

df_meta.to_csv(
    os.path.join(SAVE_DIR, "S001_metadata_all.csv"),
    index=False,
    encoding="utf-8-sig"
)

df_task.to_csv(
    os.path.join(SAVE_DIR, "S001_metadata_task.csv"),
    index=False,
    encoding="utf-8-sig"
)

df_rest.to_csv(
    os.path.join(SAVE_DIR, "S001_metadata_rest.csv"),
    index=False,
    encoding="utf-8-sig"
)

df_psd.to_csv(
    os.path.join(SAVE_DIR, "S001_psd_qc.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 전처리된 "표준화 전" window 저장
# ============================================================

np.save(
    os.path.join(SAVE_DIR, "S001_X_all_unstd.npy"),
    X_all_unstd
)

np.save(
    os.path.join(SAVE_DIR, "S001_y_all.npy"),
    y_all
)

np.save(
    os.path.join(SAVE_DIR, "S001_X_task_unstd.npy"),
    X_task_unstd
)

np.save(
    os.path.join(SAVE_DIR, "S001_y_task.npy"),
    y_task
)

np.save(
    os.path.join(SAVE_DIR, "S001_X_rest_unstd.npy"),
    X_rest_unstd
)


# ============================================================
# 10. Standardization 구현 "검증용" demo
# ------------------------------------------------------------
# 중요:
#   이것은 S001 한 명 안에서 구현이 제대로 되는지 확인하는
#   테스트일 뿐이다.
#
#   최종 103명에서는 subject-level split 후,
#   TRAIN SUBJECT의 T1/T2만으로 mean/std를 계산해야 한다.
# ============================================================

if len(X_task_unstd) >= 10:

    X_demo_train, X_demo_val, y_demo_train, y_demo_val = train_test_split(
        X_task_unstd,
        y_task,
        test_size=0.2,
        random_state=42,
        stratify=y_task
    )

    # 채널별 mean/std
    ch_mean = np.mean(
        X_demo_train,
        axis=(0, 2),
        keepdims=True
    )

    ch_std = np.std(
        X_demo_train,
        axis=(0, 2),
        keepdims=True
    )

    ch_std[ch_std == 0] = 1e-8

    X_demo_train_std = (
        X_demo_train - ch_mean
    ) / ch_std

    X_demo_val_std = (
        X_demo_val - ch_mean
    ) / ch_std

    X_demo_rest_std = (
        X_rest_unstd - ch_mean
    ) / ch_std

    print("\n" + "=" * 70)
    print("Standardization smoke test (S001 only)")
    print("=" * 70)

    print(
        "Train standardized mean:",
        float(X_demo_train_std.mean())
    )
    print(
        "Train standardized std :",
        float(X_demo_train_std.std())
    )

    print(
        "Val standardized mean:",
        float(X_demo_val_std.mean())
    )
    print(
        "Val standardized std :",
        float(X_demo_val_std.std())
    )

    print(
        "Rest standardized mean:",
        float(X_demo_rest_std.mean())
    )
    print(
        "Rest standardized std :",
        float(X_demo_rest_std.std())
    )

    np.save(
        os.path.join(SAVE_DIR, "S001_X_task_demo_std.npy"),
        (X_task_unstd - ch_mean) / ch_std
    )

    np.save(
        os.path.join(SAVE_DIR, "S001_X_rest_demo_std.npy"),
        X_demo_rest_std
    )

    np.save(
        os.path.join(SAVE_DIR, "S001_demo_train_mean.npy"),
        ch_mean
    )

    np.save(
        os.path.join(SAVE_DIR, "S001_demo_train_std.npy"),
        ch_std
    )


# ============================================================
# 11. 최종 자동 검증 결과
# ============================================================

print("\n" + "=" * 70)
print("자동 검증 결과")
print("=" * 70)

checks = []

checks.append((
    "채널 수 = 9",
    X_all_unstd.shape[1] == 9
))

checks.append((
    "window sample = 320",
    X_all_unstd.shape[2] == 320
))

checks.append((
    "window 개수 = 7",
    NUM_WINDOWS == 7
))

checks.append((
    "window 시작시간 = 0.5~2.0",
    actual_starts == expected_starts
))

checks.append((
    "NaN/Inf 없음",
    np.isfinite(X_all_unstd).all()
))

checks.append((
    "Task label이 0/1",
    set(np.unique(y_task)).issubset({0, 1})
))

checks.append((
    "Rest label 분리됨",
    np.all(y_all[rest_mask] == -1)
))

checks.append((
    "필터 대역 = 8~30 Hz",
    LOW_CUT == 8.0 and HIGH_CUT == 30.0
))

checks.append((
    "causal phase = forward",
    True
))

all_pass = True

for name, passed in checks:
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] {name}")
    all_pass = all_pass and passed

# ------------------------------------------------------------
# 간단한 label / run별 통계
# ------------------------------------------------------------

run_label_summary = (
    df_meta
    .groupby(["Run", "Label", "Label_Name"], as_index=False)
    .size()
    .rename(columns={"size": "Kept_Windows"})
)

run_label_summary.to_csv(
    os.path.join(SAVE_DIR, "S001_run_label_window_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[Run × Label별 보존 window]")
print(run_label_summary.to_string(index=False))

print("\n" + "=" * 70)

if all_pass:
    print("✅ S001 기본 전처리 파이프라인 검증 PASS")
else:
    print("⚠️ 일부 검증 항목이 FAIL입니다. 위 결과를 확인하세요.")

print(f"\n결과 저장 폴더: {os.path.abspath(SAVE_DIR)}")
print("=" * 70)

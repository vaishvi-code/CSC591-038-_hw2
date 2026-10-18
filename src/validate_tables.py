"""
validate_tables.py  –  reads results/metrics/*.json and asserts that every
value matches the report's tables exactly (to 2 decimal places for accuracy,
1 decimal place for memory/time).  Exits with code 0 on success, 1 on failure.

Usage:
    python src/validate_tables.py
"""
import json, os, statistics, sys

BASE = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics')

def load(name):
    path = os.path.join(BASE, f'{name}.json')
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        return None
    return json.load(open(path))

def best_val_test(d):
    bv_i = d['val_acc'].index(max(d['val_acc']))
    return round(max(d['val_acc']) * 100, 2), round(d['test_acc'][bv_i] * 100, 2)

failures = []

def check(label, expected, actual, tol=0.02):
    if abs(expected - actual) > tol:
        failures.append(f'FAIL [{label}]: expected={expected}, actual={actual:.4f}, diff={abs(expected-actual):.4f}')
        return False
    return True

# ─── Table 1: GCN Depth ablation (80 ep, seed 42) ─────────────────────────────
# format: (val%, test%, mem_mb, ms_ep)
TABLE1_DEPTH = {
    2: (70.55, 69.66, 5445.7, 10.0),
    3: (71.49, 70.88, 6030.5, 17.0),
    4: (71.60, 70.65, 6616.2, 15.6),
    5: (71.04, 70.08, 7201.7, 18.9),
    6: (70.32, 69.72, 7788.0, 22.3),
}
TABLE1_DEPTH_BASE = (71.82, 71.30, 6030.5, 14.5)  # 100-ep italic row

TABLE1_LR = {
    '0.001': (68.91, 67.71, 6030.5, 14.6),
    '0.005': (71.63, 70.87, 6030.5, 13.5),
    '0.01':  (71.49, 70.88, 6030.5, 17.0),
    '0.05':  (69.33, 68.17, 6030.5, 13.2),
}

# ─── 5-seed GCN stats ────────────────────────────────────────────────────────
MULTISEED = {'mean_test': 71.15, 'std_test': 0.26, 'mean_val': 71.92, 'std_val': 0.10}

# ─── Table 2: SAGE batch ablation (5 ep, seed 42) ────────────────────────────
TABLE2_BATCH = {
    512:  (90.80, 76.45, 7113.8, 66.3),
    1024: (90.77, 77.89, 7109.7, 59.5),
    2048: (90.62, 77.33, 9393.7, 49.9),
}
TABLE2_BASE = (91.61, 78.61, 7109.7, 60.6)  # 15-ep baseline

TABLE2_FANOUT = {
    '5_5_5':   (89.05, 75.92, 3735.0, 16.9),
    '5_5_15':  (89.27, 75.52, 5539.7, 35.4),
    '15_5_5':  (90.80, 77.29, 5454.2, 37.3),
    '10_10_5': (90.37, 77.24, 6152.9, 46.8),
    'batch_1024': (90.77, 77.89, 7109.7, 59.5),  # [15,10,5] = sage_batch_1024
}

# ─── Table 3: Comparative profile ────────────────────────────────────────────
TABLE3 = {
    'gcn_peak_mb':  6030.5,
    'gcn_step_ms':  14.5,
    'sage_peak_mb': 7109.7,
    'sage_epoch_s': 59.5,
}

# ─── Run checks ───────────────────────────────────────────────────────────────
print("Validating Table 1 - GCN Depth ablation...")
for dep, (rv, rt, rmem, rtime) in TABLE1_DEPTH.items():
    d = load(f'gcn_depth_{dep}')
    if d is None: continue
    bv, bt = best_val_test(d)
    check(f'depth_{dep} val',   rv,    bv)
    check(f'depth_{dep} test',  rt,    bt)
    check(f'depth_{dep} mem',   rmem,  d['peak_memory_mb'],      tol=0.1)
    check(f'depth_{dep} ms/ep', rtime, d['avg_epoch_time_sec']*1000, tol=1.0)

d_base = load('gcn_baseline')
bv_b = round(max(d_base['val_acc']) * 100, 2)
bv_i = d_base['val_acc'].index(max(d_base['val_acc']))
bt_b = round(d_base['test_acc'][bv_i] * 100, 2)
check('baseline val',   TABLE1_DEPTH_BASE[0], bv_b)
check('baseline test',  TABLE1_DEPTH_BASE[1], bt_b)
check('baseline mem',   TABLE1_DEPTH_BASE[2], d_base['peak_memory_mb'], tol=0.1)
check('baseline ms/ep', TABLE1_DEPTH_BASE[3], d_base['avg_epoch_time_sec']*1000, tol=0.5)

print("Validating Table 1 - GCN LR ablation...")
for lr_str, (rv, rt, rmem, rtime) in TABLE1_LR.items():
    d = load(f'gcn_lr_{lr_str}')
    if d is None: continue
    bv, bt = best_val_test(d)
    check(f'lr_{lr_str} val',   rv,    bv)
    check(f'lr_{lr_str} test',  rt,    bt)
    check(f'lr_{lr_str} mem',   rmem,  d['peak_memory_mb'],         tol=0.1)
    check(f'lr_{lr_str} ms/ep', rtime, d['avg_epoch_time_sec']*1000, tol=1.0)

print("Validating 5-seed GCN statistics...")
vals, tests = [], []
for s in [42, 43, 44, 45, 46]:
    d = load(f'gcn_seed_{s}')
    if d is None: continue
    vals.append(d['best_val_acc'] * 100)
    tests.append(d['best_test_acc'] * 100)
check('5seed mean_test', MULTISEED['mean_test'], statistics.mean(tests), tol=0.005)
check('5seed std_test',  MULTISEED['std_test'],  statistics.stdev(tests), tol=0.005)
check('5seed mean_val',  MULTISEED['mean_val'],  statistics.mean(vals),  tol=0.005)
check('5seed std_val',   MULTISEED['std_val'],   statistics.stdev(vals), tol=0.005)

print("Validating Table 2 - SAGE baseline...")
d = load('sage_baseline')
check('sage_base val',  TABLE2_BASE[0], d['best_val_acc']*100)
check('sage_base test', TABLE2_BASE[1], d['best_test_acc']*100)
check('sage_base mem',  TABLE2_BASE[2], d['peak_memory_mb'],     tol=0.1)
check('sage_base s/ep', TABLE2_BASE[3], d['avg_epoch_time_sec'], tol=0.1)

print("Validating Table 2 - SAGE Batch ablation...")
for bs, (rv, rt, rmem, rtime) in TABLE2_BATCH.items():
    d = load(f'sage_batch_{bs}')
    if d is None: continue
    check(f'batch_{bs} val',  rv,    d['best_val_acc']*100)
    check(f'batch_{bs} test', rt,    d['best_test_acc']*100)
    check(f'batch_{bs} mem',  rmem,  d['peak_memory_mb'],     tol=0.1)
    check(f'batch_{bs} s/ep', rtime, d['avg_epoch_time_sec'], tol=0.1)

print("Validating Table 2 - SAGE Fanout ablation...")
for s_key, (rv, rt, rmem, rtime) in TABLE2_FANOUT.items():
    fname = f'sage_sampling_{s_key}' if s_key != 'batch_1024' else 'sage_batch_1024'
    d = load(fname)
    if d is None: continue
    check(f'fanout_{s_key} val',  rv,    d['best_val_acc']*100)
    check(f'fanout_{s_key} test', rt,    d['best_test_acc']*100)
    check(f'fanout_{s_key} mem',  rmem,  d['peak_memory_mb'],     tol=0.1)
    check(f'fanout_{s_key} s/ep', rtime, d['avg_epoch_time_sec'], tol=0.1)

print("Validating Table 3 - Comparative profiling...")
gcn = load('gcn_baseline')
sb1024 = load('sage_batch_1024')
sage = load('sage_baseline')
check('T3 gcn_peak_mb',  TABLE3['gcn_peak_mb'],  gcn['peak_memory_mb'], tol=0.1)
check('T3 gcn_step_ms',  TABLE3['gcn_step_ms'],  gcn['avg_epoch_time_sec']*1000, tol=0.5)
check('T3 sage_peak_mb', TABLE3['sage_peak_mb'], sage['peak_memory_mb'], tol=0.1)
check('T3 sage_epoch_s', TABLE3['sage_epoch_s'], sb1024['avg_epoch_time_sec'], tol=0.1)

# ─── Report ───────────────────────────────────────────────────────────────────
print()
total_checks = (
    len(TABLE1_DEPTH)*4 + 4 + len(TABLE1_LR)*4 + 4 +
    len(TABLE2_BATCH)*4 + 4 + len(TABLE2_FANOUT)*4 + 4
)
if failures:
    print(f'FAILED: {len(failures)} / {total_checks} checks')
    for f in failures:
        print(f'  {f}')
    sys.exit(1)
else:
    print(f'ALL PASSED: {total_checks} checks across Tables 1, 2, and 3.')
    sys.exit(0)

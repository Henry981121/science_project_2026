"""
從 refset_top500_centroid.csv 取 Top-200，複製到新資料夾
=========================================================
原資料集保持不動。複製後寫入清單 manifest.csv 方便追蹤。
"""
import os, sys, csv, shutil, json, time
from pathlib import Path

SRC_DIR  = Path(r'C:\Users\harry\Downloads\gpt-image-2-dataset\gpt-image-2-dataset\images')
CSV_PATH = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\refset_top500_centroid.csv')
DST_DIR  = Path(r'C:\Users\harry\OneDrive\Desktop\outputs\exp_L2_gpt_image2_full\filtered_photographic_subset')
TOP_K    = 200

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[dst] {DST_DIR}')

    with open(CSV_PATH, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    rows = rows[:TOP_K]
    print(f'[load] Top-{TOP_K} from {CSV_PATH.name}')

    n_copied, n_skipped, n_missing = 0, 0, 0
    manifest = []
    t0 = time.time()
    for r in rows:
        rank = int(r['rank'])
        fn   = r['filename']
        src = SRC_DIR / fn
        new_name = f"rank{rank:03d}_{fn}"
        dst = DST_DIR / new_name

        if not src.exists():
            print(f'  [miss] {fn}')
            n_missing += 1
            continue
        if dst.exists():
            n_skipped += 1
        else:
            shutil.copy2(src, dst)
            n_copied += 1
        manifest.append({
            'rank':        rank,
            'orig_filename': fn,
            'new_filename':  new_name,
            'similarity':   r['similarity'],
            'fusion_prob':  r['fusion_prob'],
            'fusion_pred':  r['fusion_pred'],
            'clip_prob':    r.get('clip_prob', ''),
            'fft_prob':     r.get('fft_prob', ''),
            'dct_prob':     r.get('dct_prob', ''),
            'dire_prob':    r.get('dire_prob', ''),
            'noise_prob':   r.get('noise_prob', ''),
        })

    # 寫 manifest
    m_path = DST_DIR / 'manifest.csv'
    keys = ['rank', 'orig_filename', 'new_filename', 'similarity',
            'fusion_prob', 'fusion_pred',
            'clip_prob', 'fft_prob', 'dct_prob', 'dire_prob', 'noise_prob']
    with open(m_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in manifest:
            w.writerow(r)
    print(f'[saved] {m_path}')

    # 算這 200 張的準確率（驗證）
    n_AI   = sum(1 for r in manifest if r['fusion_pred'] == 'AI')
    n_Real = sum(1 for r in manifest if r['fusion_pred'] == 'Real')
    acc = n_AI / max(1, n_AI + n_Real) * 100

    summary = {
        'source_dir':   str(SRC_DIR),
        'destination':  str(DST_DIR),
        'top_k':        TOP_K,
        'n_copied':     n_copied,
        'n_skipped_existing': n_skipped,
        'n_missing_in_source': n_missing,
        'n_AI':         n_AI,
        'n_Real':       n_Real,
        'accuracy_pct': round(acc, 2),
        'wall_time_sec': round(time.time() - t0, 1),
    }
    (DST_DIR / 'copy_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print()
    print('=' * 56)
    print(f'  Top-{TOP_K} 寫實攝影風格子集 — 複製完成')
    print('=' * 56)
    print(f'  目的地             : {DST_DIR}')
    print(f'  複製成功           : {n_copied}')
    print(f'  已存在略過         : {n_skipped}')
    print(f'  原圖缺失           : {n_missing}')
    print(f'  判為 AI（正確）    : {n_AI}')
    print(f'  漏判為 Real        : {n_Real}')
    print(f'  *** 子集準確率    : {acc:.2f}% ***')
    print(f'  耗時              : {summary["wall_time_sec"]:.1f} 秒')
    print('=' * 56)


if __name__ == '__main__':
    main()

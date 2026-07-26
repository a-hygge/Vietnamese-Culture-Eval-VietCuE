"""
Gộp các per-model output files (./outputs/*.json) thành 1 file lớn:
  kept247_with_models.json — format giống tonghop/model1/van_hoa.json:
    Mỗi item có: id, category, question, answer (gold),
                 {full_name}_answer (cho mỗi model đã chạy)

Usage:
    python merge_outputs.py
    python merge_outputs.py --output ../../data/kept247_with_models.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'kept247_with_models.json')
SOURCE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'kept_items_full.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--source', default=SOURCE_FILE)
    args = parser.parse_args()

    with open(args.source, encoding='utf-8') as f:
        source = json.load(f)
    print(f'Source: {len(source)} items')

    by_id = {item['id']: dict(item) for item in source}

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*.json')))
    print(f'Found {len(files)} per-model files:')

    models_added = []
    for fp in files:
        with open(fp, encoding='utf-8') as f:
            data = json.load(f)
        meta = data.get('metadata', {})
        full_name = meta.get('full_name')
        if not full_name:
            print(f'  ⚠️  Skipped (no full_name): {os.path.basename(fp)}')
            continue
        col_name = f'{full_name}_answer'
        details = data.get('details', [])
        ok_count = 0
        err_count = 0
        for r in details:
            sid = r['id']
            if sid not in by_id:
                continue                                   
            if r.get('answer') and not r.get('error'):
                by_id[sid][col_name] = r['answer']
                ok_count += 1
            else:
                                                    
                by_id[sid][col_name] = r.get('answer', '')
                err_count += 1
        models_added.append((full_name, ok_count, err_count, len(details)))
        print(f'  ✓ {full_name}: {ok_count} OK, {err_count} err / {len(details)} total')

    merged = [by_id[item['id']] for item in source]

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'\n✅ Saved: {args.output}  ({size_mb:.2f} MB)')

    print(f'\n=== SUMMARY ===')
    print(f'Items: {len(merged)}')
    print(f'Models in merged file:')
    sample_keys = [k for k in merged[0].keys() if k.endswith('_answer')]
    for k in sample_keys:
        non_empty = sum(1 for item in merged if item.get(k))
        print(f'  {k:<55s} {non_empty}/{len(merged)} non-empty')

if __name__ == '__main__':
    main()

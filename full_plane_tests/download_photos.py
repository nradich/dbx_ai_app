"""Commons acquisition and review utilities; run from the repository root.

No inference or training is performed. All generated paths are within this folder.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent
CLASSES = ['alaska', 'american', 'delta', 'southwest', 'united']
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'AirlinePhotoReview/1.0 (educational image attribution and review)'
API = 'https://commons.wikimedia.org/w/api.php'


def clean(value):
    return html.unescape(re.sub('<[^>]+>', '', value or '')).strip()


def get(url, **kwargs):
    for attempt in range(5):
        response = SESSION.get(url, timeout=120, **kwargs)
        if response.status_code in (429, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response
    response.raise_for_status()


def discover():
    sources = '\n'.join(p.read_text(encoding='utf-8') for p in (ROOT.parent / 'airline_tails').glob('*/SOURCES.md'))
    excluded_titles = {unquote(t).replace('_', ' ').removeprefix('File:') for t in re.findall(r'https://commons.wikimedia.org/wiki/File:([^\s)]+(?:\)[^\s)]*)?)', sources)}
    excluded_regs = set(re.findall(r'\bN[0-9][A-Z0-9]{2,5}\b', sources))
    candidates = []
    for airline in CLASSES:
        data = get(API, params={'action': 'query', 'format': 'json', 'generator': 'search', 'gsrsearch': f'"{airline.title()} Airlines" Boeing 737 filetype:bitmap', 'gsrnamespace': 6, 'gsrlimit': 35, 'prop': 'imageinfo', 'iiprop': 'url|size|extmetadata|sha1', 'iiurlwidth': 900}).json()
        for page in sorted(data.get('query', {}).get('pages', {}).values(), key=lambda p: p.get('index', 0)):
            info = page['imageinfo'][0]
            title = page['title']
            meta = info.get('extmetadata', {})
            text = title + ' ' + clean(meta.get('ImageDescription', {}).get('value', ''))
            regs = set(re.findall(r'\bN[0-9][A-Z0-9]{2,5}\b', text))
            if info['width'] < 1600 or regs & excluded_regs or title.removeprefix('File:') in excluded_titles:
                continue
            license_name = clean(meta.get('LicenseShortName', {}).get('value', ''))
            if not license_name.startswith(('CC BY', 'CC0', 'Public domain')):
                continue
            record = {'id': f'{airline}_{len([r for r in candidates if r["airline"] == airline]) + 1:02}', 'airline': airline, 'title': title, 'source_url': info['descriptionurl'], 'original_url': info['url'], 'preview_url': info.get('thumburl', info['url']), 'artist': clean(meta.get('Artist', {}).get('value', '')), 'license': license_name, 'license_url': meta.get('LicenseUrl', {}).get('value', ''), 'source_width': info['width'], 'source_height': info['height'], 'commons_sha1': info['sha1'], 'registrations': sorted(regs)}
            candidates.append(record)
            if len([r for r in candidates if r['airline'] == airline]) >= 8:
                break
        print(airline, len([r for r in candidates if r['airline'] == airline]), flush=True)
    (ROOT / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n', encoding='utf-8')
    preview = ROOT / 'review' / 'candidates'
    preview.mkdir(parents=True, exist_ok=True)
    for r in candidates:
        dest = preview / (r['id'] + '.jpg')
        if not dest.exists():
            im = Image.open(io.BytesIO(get(r['preview_url']).content)).convert('RGB')
            im.save(dest, quality=90)
    for airline in CLASSES:
        rows = [r for r in candidates if r['airline'] == airline]
        sheet = Image.new('RGB', (1200, ((len(rows) + 1) // 2) * 350), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(rows):
            im = Image.open(preview / (r['id'] + '.jpg'))
            im.thumbnail((590, 300))
            x, y = i % 2 * 600, i // 2 * 350
            sheet.paste(im, (x, y + 30))
            draw.text((x + 5, y + 5), f'{r["id"]} {r["source_width"]}x{r["source_height"]}', fill='black')
        sheet.save(ROOT / 'review' / f'candidates_{airline}.jpg', quality=90)


def download():
    candidates = json.loads((ROOT / 'candidates.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    records = []
    for choice in selection:
        r = next(r.copy() for r in candidates if r['id'] == choice['candidate'])
        r.update({k: v for k, v in choice.items() if k != 'candidate'})
        preview_url = r.pop('preview_url')
        destination = ROOT / 'images' / r['airline'] / (r['id'] + '.jpg')
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = SESSION.get(r['original_url'], timeout=120)
        if response.status_code == 429:
            # Commons recommends standard-size thumbnails when originals are throttled.
            download_url = re.sub(r'/960px-', '/1920px-', preview_url).split('?')[0]
            raw = get(download_url).content
            r['download_kind'] = 'Commons 1920px derivative (original throttled)'
        else:
            response.raise_for_status()
            raw = response.content
            download_url = r['original_url']
            r['download_kind'] = 'original'
        r['download_url'] = download_url
        r['original_url'] = r['original_url'].split('?')[0]
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
        assert image.width >= 1600
        r['source_sha256'] = hashlib.sha256(raw).hexdigest()
        r['downloaded_source_sha1'] = hashlib.sha1(raw).hexdigest()
        r['downloaded_source_width'], r['downloaded_source_height'] = image.size
        if r['download_kind'] == 'original':
            assert r['downloaded_source_sha1'] == r['commons_sha1']
        image.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
        image = Image.frombytes('RGB', image.size, image.tobytes())
        image.save(destination, quality=90, optimize=True)
        r.update(path=destination.relative_to(ROOT).as_posix(), width=image.width, height=image.height, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), downloaded=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        records.append(r)
        print(r['id'], image.size, flush=True)
        time.sleep(2)
    (ROOT / 'manifest.json').write_text(json.dumps({'version': 1, 'classes': CLASSES, 'bbox_format': 'normalized [left, top, right, bottom] in saved oriented full image; manual vertical-tail region including a small margin', 'selection_policy': 'External photographs selected and manually reviewed before inference; no model access or training.', 'independence_note': 'No strict independence from ImageNet or other pretraining data is assured.', 'records': records}, indent=2) + '\n', encoding='utf-8')


def finalize():
    # Disable bytecode so importing the existing helper does not write outside ROOT.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('prepare_data', ROOT.parent / 'training' / 'prepare_data.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    choices = {r['id']: r for r in json.loads((ROOT / 'selection.json').read_text())}
    for r in manifest['records']:
        r['bbox'] = choices[r['id']]['bbox']
        r['review_note'] = choices[r['id']]['review_note']
        r['manual_bbox_reviewed'] = True
    annotations = json.loads((ROOT.parent / 'dataset' / 'tail_annotations.json').read_text())
    existing = []
    for r in annotations['records']:
        path = ROOT.parent / 'airline_tails' / r['source_path']
        im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        existing.append((r['source_path'], hashlib.sha256(path.read_bytes()).hexdigest(), {'source': helpers.fingerprints(im), 'crop': helpers.fingerprints(helpers.crop_tail(im, r['bbox']))}))
    findings = []
    new = []
    sheet = Image.new('RGB', (1600, 5 * 380), 'white')
    crop_sheet = Image.new('RGB', (1200, 5 * 280), 'white')
    draw, crop_draw = ImageDraw.Draw(sheet), ImageDraw.Draw(crop_sheet)
    for i, r in enumerate(manifest['records']):
        im = Image.open(ROOT / r['path']).convert('RGB')
        crop = helpers.crop_tail(im, r['bbox'])
        fp = {'source': helpers.fingerprints(im), 'crop': helpers.fingerprints(crop)}
        r['fingerprints'] = fp
        for name, source_hash, old in existing + new:
            for view in ('source', 'crop'):
                for old_view in ('source', 'crop'):
                    a, b = fp[view], old[old_view]
                    d, p = (a['dhash'] ^ b['dhash']).bit_count(), (a['phash'] ^ b['phash']).bit_count()
                    if a['pixels'] == b['pixels'] or (d <= 4 and p <= 8) or r['source_sha256'] == source_hash:
                        findings.append({'id': r['id'], 'existing': name, 'view': view, 'existing_view': old_view, 'dhash_distance': d, 'phash_distance': p})
        new.append((r['id'], r['source_sha256'], fp))
        preview = im.copy()
        preview.thumbnail((790, 330))
        x, y = i % 2 * 800, i // 2 * 380
        sheet.paste(preview, (x, y + 30))
        draw.text((x + 5, y + 5), r['id'], fill='black')
        l, t, right, b = r['bbox']
        draw.rectangle((x + l * preview.width, y + 30 + t * preview.height, x + right * preview.width, y + 30 + b * preview.height), outline='red', width=2)
        crop.thumbnail((580, 240))
        x, y = i % 2 * 600, i // 2 * 280
        crop_sheet.paste(crop, (x, y + 25))
        crop_draw.text((x + 5, y + 5), r['id'], fill='black')
    (ROOT / 'review').mkdir(exist_ok=True)
    sheet.save(ROOT / 'review' / 'full_planes.jpg', quality=90)
    crop_sheet.save(ROOT / 'review' / 'tail_crops.jpg', quality=90)
    manifest['duplicate_checks'] = {'method': 'training.prepare_data fingerprints: exact source SHA256, decoded RGB, dHash <=4/64 AND pHash <=8/63; all full/crop combinations against 50 local source photos and manual crops, plus within this set.', 'existing_photos_checked': len(existing), 'findings': findings, 'limitations': 'Heuristic screen, not proof of independent capture or absence from pretraining. Original remote full sources for existing cropped photos were not re-downloaded.'}
    (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    lines = ['# Full-plane photograph attribution', '', 'Wikimedia Commons photographs; each image retains its individual license. Public creator credits below are intentional attribution. Saved derivatives are resized, JPEG re-encoded, and stripped of inherited metadata; no full-plane framing was cropped. Review tail crops are manual cropped derivatives under the same respective licenses.', '', '| ID / saved file | Original source | Creator | License | Original → saved resolution | Manual bbox (L,T,R,B) |', '|---|---|---|---|---|---|']
    for r in manifest['records']:
        lines.append(f'| [{r["id"]}]({r["path"]}) | [{r["title"]}]({r["source_url"]}) | {r["artist"].replace("|", "/")} | [{r["license"]}]({r["license_url"]}) | {r["source_width"]}×{r["source_height"]} → {r["width"]}×{r["height"]} | {r["bbox"]} |')
    (ROOT / 'SOURCES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Discard generated search previews: the retained dataset has exactly ten photos.
    for path in (ROOT / 'review').glob('candidates_*.jpg'):
        path.unlink()
    previews = ROOT / 'review' / 'candidates'
    if previews.exists():
        for path in previews.glob('*.jpg'):
            path.unlink()
        previews.rmdir()
    print(json.dumps({'records': len(manifest['records']), 'duplicate_findings': findings}, indent=2))


def validate():
    from collections import Counter
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    records = manifest['records']
    assert manifest['classes'] == CLASSES
    assert Counter(r['airline'] for r in records) == Counter({c: 2 for c in CLASSES})
    sources = '\n'.join(p.read_text(encoding='utf-8') for p in (ROOT.parent / 'airline_tails').glob('*/SOURCES.md'))
    excluded_regs = set(re.findall(r'\bN[0-9][A-Z0-9]{2,5}\b', sources))
    assert len({reg for r in records for reg in r['registrations']}) == 10
    assert len({r['sha256'] for r in records}) == 10
    assert len(list((ROOT / 'images').glob('*/*.jpg'))) == 10
    for r in records:
        assert r['source_url'] not in sources
        assert not set(r['registrations']) & excluded_regs
        assert r['source_width'] >= 1600
        path = ROOT / r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
        with Image.open(path) as im:
            assert im.format == 'JPEG' and im.mode == 'RGB'
            assert im.size == (r['width'], r['height']) and max(im.size) <= 1920
            assert not im.getexif() and not any(k in im.info for k in ('exif', 'icc_profile', 'comment', 'xmp'))
        left, top, right, bottom = r['bbox']
        assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
    assert not manifest['duplicate_checks']['findings']
    print(json.dumps({'validation': 'passed', 'counts': dict(Counter(r['airline'] for r in records)), 'download_kinds': dict(Counter(r['download_kind'] for r in records)), 'photos': [{k: r[k] for k in ('id', 'source_width', 'source_height', 'width', 'height', 'bbox')} for r in records]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['discover', 'download', 'finalize', 'validate'])
    globals()[parser.parse_args().action]()

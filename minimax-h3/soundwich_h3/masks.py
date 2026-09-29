"""SAM prompt policy from the scene and alignment of SAM masks to H3's packed video-token grid.

Each stem declares either `sam_prompt` (a text prompt, or a list whose masks
are unioned, e.g. performer plus instrument), or `video_routing`:
`background_only` (owner = complement of all foreground entity masks) or
`offscreen` (no owner video tokens). A stem with a single text prompt may add
reviewed click points: `sam_points` (normalized x/y, positive or negative
label) on source frame `sam_point_frame`.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MASK_THRESHOLD = .15


def token_grid(scene):
    """(latent frames, token rows, token columns) of H3's packed video for this geometry."""
    frames, width, height = scene['num_frames'], scene['width'], scene['height']
    if (frames - 5) % 17 or width % 32 or height % 32:
        raise ValueError('Expected num_frames = 17n+5 and width/height multiples of 32')
    return (frames - 5) // 17 * 5 + 2, height // 32, width // 32


def latent_frame_centers(num_latents, num_frames):
    """Representative source frames for H3's 17-frame/5-latent VAE chunks.

    Native RoPE uses frame spans (1,4,4,4,4) per chunk, so offsets 0,3,7,11,15
    represent the five successive latent bins. This is an alignment proxy:
    VAE temporal mixing can read beyond the sampled frame.
    """
    offsets = (0, 3, 7, 11, 15)
    return [min((i // 5) * 17 + offsets[i % 5], num_frames - 1) for i in range(num_latents)]


def _prompt_key(text):
    # The SAM backend collapses whitespace and appends a period; compare prompts the same way.
    return ' '.join(str(text).split()).rstrip('.')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_policy(scene):
    prompts = {}
    owners = []
    for stem in scene['stems']:
        routing = stem.get('video_routing')
        texts = stem.get('sam_prompt') or []
        texts = [texts] if isinstance(texts, str) else list(texts)
        if bool(texts) == bool(routing):
            raise ValueError(f'Stem {stem["id"]!r} needs exactly one of sam_prompt or video_routing')
        if routing:
            if routing not in ('background_only', 'offscreen'):
                raise ValueError(f'Unknown video_routing for {stem["id"]!r}: {routing!r}')
            owners.append(dict(stem_id=stem['id'], components=[],
                               policy='background_only' if routing == 'background_only' else 'offscreen_audio_only'))
            continue
        points = _sam_points(stem)
        if points and len(texts) != 1:
            raise ValueError(f'Stem {stem["id"]!r}: sam_points need exactly one sam_prompt')
        components = []
        for text in texts:
            key = (str(text).strip(), json.dumps(points, sort_keys=True) if points else '')
            if key not in prompts:
                prompts[key] = dict(id=f'entity_{len(prompts):02d}', prompt=key[0], **points)
            components.append(prompts[key]['id'])
        owners.append(dict(stem_id=stem['id'], components=components))
    if not prompts:
        raise ValueError('Entity routing needs at least one stem with a sam_prompt')
    return dict(case_id=scene['id'], prompts=list(prompts.values()), owners=owners)


def _sam_points(stem):
    raw = stem.get('sam_points') or []
    if not raw:
        return {}
    if stem.get('sam_point_frame') is None:
        raise ValueError(f'Stem {stem["id"]!r}: sam_points require sam_point_frame')
    points = []
    for item in raw:
        x, y, label = float(item['x']), float(item['y']), str(item.get('label', 'positive')).lower()
        if not (0 <= x <= 1 and 0 <= y <= 1) or label not in ('positive', 'negative'):
            raise ValueError(f'Stem {stem["id"]!r}: invalid sam point {item!r}')
        points.append(dict(x=x, y=y, label=label))
    return dict(points=points, point_frame_index=int(stem['sam_point_frame']))


def backend_prompt_spec(policy):
    keys = ('id', 'prompt', 'points', 'point_frame_index')
    return dict(case_id=policy['case_id'], stems=[{k: p[k] for k in keys if k in p} for p in policy['prompts']])


def load_entity_masks(scene, root, policy, *, threshold=MASK_THRESHOLD):
    """Full-resolution SAM masks -> boolean [stem, packed video token] ownership."""
    root = Path(root)
    backend = json.loads((root/'sam_masks.json').read_text())
    if backend.get('case_id') != scene['id'] or backend.get('frame_count') != scene['num_frames']:
        raise ValueError(f'SAM case/frame mismatch: {root}')
    if backend.get('sampled_frame_indices') != list(range(scene['num_frames'])):
        raise ValueError('H3 requires SAM masks for every source frame')
    if [p['id'] for p in policy['prompts']] != list(backend['stems']):
        raise ValueError('SAM prompt/component order does not match policy')
    for p in policy['prompts']:
        if _prompt_key(backend['stems'][p['id']]['prompt']) != _prompt_key(p['prompt']):
            raise ValueError(f'SAM prompt mismatch: {p["id"]}')
    frames, rows, cols = token_grid(scene)
    video_tokens = frames * rows * cols
    centers = latent_frame_centers(frames, scene['num_frames'])
    component, coverage_by_component = {}, {}
    for item in policy['prompts']:
        name, label = item['id'], item['prompt']
        directory = root/'masks'/name
        if len(list(directory.glob('frame_*.png'))) != scene['num_frames']:
            raise ValueError(f'Incomplete SAM masks for prompt {label!r}: {directory}')
        grids, fractions = [], []
        for frame in centers:
            image = Image.open(directory/f'frame_{frame:05d}.png').convert('L')
            if image.size != (scene['width'], scene['height']):
                raise ValueError(f'SAM mask resolution mismatch for prompt {label!r}: {directory}, frame {frame}')
            fractions.append(float((np.asarray(image) > 127).mean()))
            grids.append(np.asarray(image.resize((cols, rows), Image.Resampling.BOX), dtype=np.float32) / 255.)
        mean = float(np.mean(fractions))
        if not .001 < mean < .85:
            raise ValueError(f'Empty or implausible SAM mask for prompt {label!r} ({mean:.5f}): {directory}')
        component[name] = np.stack(grids)
        coverage_by_component[name] = mean
    foreground = np.zeros((frames, rows, cols), dtype=bool)
    for owner in policy['owners']:
        if owner.get('policy') != 'background_only':
            for name in owner['components']:
                foreground |= component[name] >= threshold
    masks = []
    for owner in policy['owners']:
        if owner.get('policy') == 'background_only':
            mask = ~foreground
            if not mask.any():
                raise ValueError('No background tokens remain for a background_only stem')
            masks.append(mask.reshape(-1))
            continue
        if not owner['components']:
            masks.append(np.zeros(video_tokens, dtype=bool))
            continue
        mask = np.maximum.reduce([component[name] for name in owner['components']]) >= threshold
        coverage = float(mask.mean())
        if not .002 < coverage < .85:
            raise ValueError(f'Empty or implausible owner mask for {owner["stem_id"]} ({coverage:.5f})')
        masks.append(mask.reshape(-1))
    result = np.stack(masks)
    metadata = dict(grid=[frames, rows, cols], frame_centers=centers, threshold=threshold,
                    component_mean_fraction=coverage_by_component,
                    stem_fraction=[float(row.mean()) for row in result],
                    packed_owner_sha256=hashlib.sha256(result.tobytes()).hexdigest())
    return torch.from_numpy(result), metadata

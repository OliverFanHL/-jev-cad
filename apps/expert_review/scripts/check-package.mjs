import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = fileURLToPath(new URL('../', import.meta.url));
const manifest = JSON.parse(fs.readFileSync(new URL('../data/pairs.json', import.meta.url), 'utf8'));
if (!manifest.pairs.length || !manifest.asset_ids.length) {
  throw new Error('The pair manifest requires pairs and CAD assets');
}
const assets = new Set(manifest.asset_ids);
const referenced = new Set(manifest.pairs.flatMap(pair => [pair.query_id, pair.candidate_id]));
if (assets.size !== manifest.asset_ids.length || assets.size !== referenced.size ||
    [...referenced].some(id => !assets.has(id))) {
  throw new Error('Pair references and unique CAD assets must agree');
}
const ids = new Set(manifest.pairs.map(p => p.pair_id));
if (ids.size !== manifest.pairs.length) throw new Error('Duplicate pair ID');
const missing = manifest.asset_ids.filter(id => ['', '_B'].some(suffix =>
  !fs.existsSync(path.join(root, 'public', 'renders', `${id}${suffix}.webp`))));
if (missing.length) throw new Error(`${missing.length} CAD assets lack a required view; run scripts/prepare_package.py --render before deployment`);
const provenance = JSON.parse(fs.readFileSync(new URL('../data/render_provenance.json', import.meta.url), 'utf8'));
if (provenance.protocol_hash !== manifest.protocol_hash ||
    Object.keys(provenance.assets).length !== manifest.asset_ids.length) {
  throw new Error('Rendered asset provenance differs from the frozen protocol');
}
for (const id of manifest.asset_ids) {
  for (const [suffix, key] of [['', 'view_a_sha256'], ['_B', 'view_b_sha256']]) {
    const bytes = fs.readFileSync(path.join(root, 'public', 'renders', `${id}${suffix}.webp`));
    const digest = createHash('sha256').update(bytes).digest('hex');
    if (digest !== provenance.assets[id]?.[key]) throw new Error(`Render checksum differs for ${id}${suffix}`);
  }
}
console.log(`Verified ${manifest.pairs.length} pairs and two views of ${manifest.asset_ids.length} blinded CAD assets.`);

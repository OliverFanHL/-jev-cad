import fs from 'node:fs';

export const manifest = JSON.parse(fs.readFileSync(new URL('../data/pairs.json', import.meta.url), 'utf8'));
export const pairById = new Map(manifest.pairs.map(pair => [pair.pair_id, pair]));

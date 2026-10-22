# CAD pair review app

A browser interface for comparing two mechanical parts and recording relevance
grades from 0 to 3. Two reviewers work independently; an adjudicator resolves
disagreements. Each part appears under an opaque ID with two rendered views.

The app uses Node.js 20 or later, a PostgreSQL database, and separate access
tokens for the two reviewers, adjudicator, and export role.

## Prepare the CAD pairs

First create the candidate pool and annotation files using the
[main workflow](../../docs/REPRODUCTION.md). From the repository root:

```bash
python apps/expert_review/scripts/prepare_package.py --render --workers 4
```

This reads `outputs/study/full/annotations/`, writes the blinded manifest to
`apps/expert_review/data/pairs.json`, and creates two WebP views per part under
`public/renders/`. Rendering needs locally available STEP files, CadQuery, VTK,
and Pillow. Use `--limit 3` for a small rendering check; omit `--limit` to build
the complete package.

If starting with a saved artifact bundle, expand it first and update its CAD
paths with `python scripts/materialize_artifacts.py --dataset /path/to/TMCAD`.
The repository includes the app source; manifests, renders, and ratings are
generated locally.

## Set up the app

From this directory:

```bash
npm install
npm test
npm run check
```

`npm test` checks the grading and export logic against the prepared manifest.
`npm run check` also verifies the complete render package and its hashes. The
current checks use the recorded study selection of 6,645 pairs and 2,643 parts.

Use [.env.example](.env.example) to set `DATABASE_URL` and the four private tokens
in your deployment environment. Initialize the database with:

```bash
npm run db:init
```

`vercel.json` configures the static interface and API routes for Vercel. Deploy
this app directory with its blinded manifest and renders. Keep the source class
labels and private asset mapping in the analysis workspace.

## Export grades

Use the export role to download each reviewer's CSV and the adjudication CSV.
Check an export from this directory with:

```bash
python scripts/validate_export.py /path/to/rater_a.csv --role rater_a
```

Copy the validated exports to the annotation directory as `rater_a.csv`,
`rater_b.csv`, and `adjudication.csv`. Then run `study-grade-finalize` from the
repository root with the same study configuration to build the retrieval grades.

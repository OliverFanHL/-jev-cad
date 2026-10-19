# Blinded annotation application source

This optional application implements the independent-rater and adjudication
workflow. It is supplied as reusable protocol software. The packaged rater CSV
files are blank forms; app source is not evidence of completed assessments.

First expand the repository artifacts and rebase a locally obtained CAD dataset:

```bash
python scripts/materialize_artifacts.py --dataset /your/local/TMCAD
python research_v2/expert_review_app/scripts/prepare_package.py --render --workers 4
```

The rendering stage needs CadQuery, VTK and Pillow and generates two views per
opaque asset ID. It checks the frozen pair identities and produces local
`public/renders/` assets and `data/render_provenance.json`; those rebuildable
assets are not committed here.

Within this app directory, install Node dependencies and run the existing tests:

```bash
npm install
npm test
npm run check
```

Database-backed execution requires a provisioned PostgreSQL database and four
distinct private tokens. `.env.example` contains placeholders only. The schema
is `schema.sql`; with your own `DATABASE_URL`, `npm run db:init` initializes it.
The source includes the API handlers and static interface. It is not deployed
or connected to an existing study database by this release.

Deploy only this app package and its blinded renders for independent rating.
The larger analysis repository contains class labels and source mappings.
Exported real ratings can be validated with `scripts/validate_export.py` and
the main study's finalization code. Do not fill or export synthetic grades as
expert evidence.

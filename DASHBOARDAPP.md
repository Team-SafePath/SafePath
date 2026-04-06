# SafePath Dashboard App

This file explains how to run the **Next.js dashboard app** for SafePath and avoid the most common setup issues.

## Dashboard location

The dashboard lives in:

```text
SafePath/safepath-dashboard/
```

The main app structure is:

```text
safepath-dashboard/
  app/
  public/
    data/
  src/
    components/
```

## Requirements

Before running the app, make sure you have:

- Node.js 18+ installed
- npm installed
- the SafePath project cloned locally
- dashboard data exports available in `public/data/`

## Install dependencies

From the `safepath-dashboard` directory:

```bash
npm install
```

If dependencies are missing or the app was just pulled fresh, this command should be run first.

## Run the dashboard

From:

```text
SafePath/safepath-dashboard/
```

run:

```bash
npm run dev
```

Then open:

```text
http://localhost:3000
```

## Main routes

The dashboard currently uses these routes:

- `/` → Home
- `/crash-map` → Interactive NYC crash/risk map
- `/insights` → Why / Where / When insights page

## Required data files

These files should exist inside:

```text
safepath-dashboard/public/data/
```

### Required now

```text
lightgbm_full_panel_feature_importance.csv
segment_combined_map.geojson
hmm_daily_states.csv
```

### Optional / only needed if referenced by components

```text
segment_clusters.csv
lightgbm_full_panel_metrics.json
```

## How to copy data into the dashboard

From the main `SafePath` project root, copy files like this:

```bash
mkdir -p safepath-dashboard/public/data

cp models/lightgbm_full_panel_feature_importance.csv safepath-dashboard/public/data/
cp dashboard_exports/segment_combined_map.geojson safepath-dashboard/public/data/
cp data/processed/hmm_daily_states.csv safepath-dashboard/public/data/
cp data/processed/segment_clusters.csv safepath-dashboard/public/data/
```

Only copy files that the current dashboard actually uses.

## Important app conventions

### 1. Pages go in `app/`

Examples:

```text
app/page.tsx
app/crash-map/page.tsx
app/insights/page.tsx
```

### 2. Static data goes in `public/data/`

Examples:

```text
public/data/segment_combined_map.geojson
public/data/lightgbm_full_panel_feature_importance.csv
```

These are loaded in the browser using:

```ts
fetch("/data/filename.csv")
```

### 3. Reusable UI components go in `src/components/`

Examples:

```text
src/components/CrashMap.tsx
src/components/FeatureImportanceChart.tsx
src/components/TopRiskSegments.tsx
src/components/RiskTimingSummary.tsx
```

## Important Next.js / Leaflet note

The crash map uses `react-leaflet`, which requires the browser.

Because of that:

- the crash map page must be a **client component**
- the map component must be loaded with a **dynamic import with `ssr: false`**

Example pattern used in `app/crash-map/page.tsx`:

```tsx
"use client";

import dynamic from "next/dynamic";

const CrashMap = dynamic(() => import("@/src/components/CrashMap"), {
  ssr: false,
});
```

If this pattern is not used, you may get errors like:

- `window is not defined`
- ``ssr: false` is not allowed with `next/dynamic` in Server Components`

## Common troubleshooting

### Problem: map page crashes with `window is not defined`

Fix:

- make sure the page has:

```tsx
"use client";
```

- and that the map is imported with:

```tsx
const CrashMap = dynamic(() => import("@/src/components/CrashMap"), {
  ssr: false,
});
```

### Problem: data file does not load

Check that the file is actually in:

```text
safepath-dashboard/public/data/
```

Then test it directly in the browser, for example:

```text
http://localhost:3000/data/segment_combined_map.geojson
```

If the browser cannot open the file directly, the app will not be able to fetch it either.

### Problem: route validator / Next generated files look strange

Do not edit generated Next.js validator files manually.

Instead:

1. save your route files
2. stop the dev server
3. rerun:

```bash
npm run dev
```

### Problem: chart TypeScript tooltip formatter errors

`recharts` tooltip values may be `number | string | undefined`.

Avoid forcing strict `number` types directly in tooltip callbacks. Cast safely instead.

### Problem: map feels laggy

That is usually caused by rendering a large GeoJSON in the browser.

The fix is not deployment alone. Typical fixes are:

- simplify geometry
- reduce file size
- filter to top-risk segments only
- limit map layers shown at once

## Recommended startup sequence

Every time you come back to the project:

1. go to the dashboard folder

```bash
cd safepath-dashboard
```

2. make sure dependencies are installed

```bash
npm install
```

3. confirm required files exist in `public/data/`

4. run the dev server

```bash
npm run dev
```

5. test these pages:

```text
http://localhost:3000
http://localhost:3000/crash-map
http://localhost:3000/insights
```

## If you pull new dashboard code from GitHub

After pulling:

```bash
git pull
cd safepath-dashboard
npm install
npm run dev
```

If data files are not tracked in GitHub because of size limits, re-copy them into `public/data/` before running the app.

## Suggested note for collaborators

If someone else is running this dashboard, tell them:

- do not expect large modeling outputs to be in GitHub
- the dashboard depends on small exported data files placed into `public/data/`
- if a page is blank, first verify the needed file is present in `public/data/`

## Current dashboard focus

The app is currently organized around:

- **Crash Map** → interactive map for historical crashes and predicted risk
- **Insights** → Why / Where / When crash analysis

## Future improvements

Potential next steps:

- simplify map GeoJSON for better performance
- add date-based predicted risk filtering
- add HMM regime overlays or date badges
- polish card spacing and typography
- deploy to Vercel once data exports are finalized

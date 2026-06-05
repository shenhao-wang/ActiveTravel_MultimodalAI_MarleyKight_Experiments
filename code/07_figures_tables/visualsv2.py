# Generated from: visualsv2.ipynb
# Converted at: 2026-05-19T02:33:18.661Z

"""
Tampa Active Travel — Shapefile-Based Maps
==========================================
Produces choropleth maps using actual polygon geometries.

Zoning polygons (Zoning_District.shp) — full polygon rendering always available.
Block group polygons (tl_2025_12_bg.shp) — used if present in the working
  directory; falls back to centroid scatter plots if the file is missing.

Required in working directory:
  Zoning_District.shp / .dbf / .shx / .prj   (included)
  tampatrips_1_.csv
  tpa_zon_geoid.csv
  zone_features_combined.csv

Optional (drop in same folder to unlock full block-group polygon maps):
  tl_2025_12_bg.shp

Outputs:
  map1_zone_context.png       — Zoning polygons coloured by residential/commercial/other
  map2_zone_class.png         — Top zone classes (individual colours)
  map3_bg_entropy.png         — Block group zoning entropy choropleth
  map4_bg_ped_score.png       — Block group pedestrian orientation score
  map5_bg_at_rate.png         — Block group active travel rate (origin)
  map6_combined.png           — 2×3 panel combining all maps
"""

import os, struct, math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ─────────────────────────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────────────────────────
RESIDENTIAL = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO','RO-1','SH-RS','SH-RS-A','SH-RM','SH-RO','SH-PD',
    'YC-2','YC-4','YC-8','YC-9',
]
NON_RESIDENTIAL = [
    'CG','CI','CN','OP','OP-1','IG','IH','CBD-1','CBD-2',
    'CD-1','CD-2','CD-3','NMU-35','NMU-24','NMU-16',
    'SH-CG','SH-CI','SH-CN','YC-1','YC-3','YC-5','YC-6','YC-7',
]

# Green/orange/grey: standard land-use map convention (green = residential,
# orange = commercial/industrial). Consistent with Tampa comp plan maps.
RES_COLOR  = '#66BB6A'
NRE_COLOR  = '#FFA726'
OTH_COLOR  = '#BDBDBD'
AT_COLOR   = '#2196F3'

# Map background
WATER_COLOR = '#AED6F1'   # soft blue  — Tampa Bay / water
LAND_COLOR  = '#F5F0E8'   # warm parchment — land background
BG_COLOR    = LAND_COLOR

# Distinct colours for top individual zone classes
ZONE_PALETTE = {
    'RS-60':  '#1565C0', 'RS-50':  '#1976D2', 'RS-75':  '#2196F3',
    'RS-100': '#42A5F5', 'RS-150': '#90CAF9',
    'PD':     '#9C27B0', 'PD-A':   '#CE93D8',
    'CG':     '#E65100', 'CI':     '#F57C00', 'CN':     '#FFA726',
    'RM-16':  '#2E7D32', 'RM-24':  '#43A047', 'RM-18':  '#66BB6A',
    'IH':     '#B71C1C', 'IG':     '#E53935',
    'SH-RS':  '#00838F', 'SH-CG':  '#FF8F00',
    'CBD-2':  '#4A148C', 'CBD-1':  '#6A1B9A',
    'OTHER':  '#EEEEEE',
}

# ─────────────────────────────────────────────────────────────
# 1. SHP/DBF READERS (pure Python — no geopandas needed)
# ─────────────────────────────────────────────────────────────
def merc_to_wgs84(x, y):
    """Convert Web Mercator (EPSG:3857) to WGS84 lon/lat."""
    lon = math.degrees(x / 6378137.0)
    lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
    return lon, lat

def read_dbf(path):
    """Read a DBF file, return list of dicts."""
    with open(path, 'rb') as f:
        header = f.read(32)
        n_records   = struct.unpack('<I', header[4:8])[0]
        header_size = struct.unpack('<H', header[8:10])[0]
        record_size = struct.unpack('<H', header[10:12])[0]
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D or len(fd) < 32:
                break
            name  = fd[:11].replace(b'\x00', b'').decode('ascii', 'ignore').strip()
            ftype = chr(fd[11])
            flen  = fd[16]
            fields.append((name, ftype, flen))
        f.seek(header_size)
        rows = []
        for _ in range(n_records):
            rec = f.read(record_size)
            if not rec or rec[0] == 0x1A:
                break
            row = {}
            pos = 1
            for name, ftype, flen in fields:
                val = rec[pos:pos + flen].decode('ascii', 'ignore').strip()
                row[name] = val
                pos += flen
            rows.append(row)
    return rows

def read_shp_polygons(shp_path, is_mercator=True):
    """
    Read a polygon shapefile (type 5).
    Returns list of dicts: {bbox, parts: [array of (lon,lat) tuples]}
    Null/empty shapes are kept as empty records to preserve DBF alignment.
    """
    NULL = {'bbox': None, 'parts': [], 'ZONECLASS': '', 'ZONEDESC': ''}
    records = []
    with open(shp_path, 'rb') as f:
        f.read(100)  # file header
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            content_len = struct.unpack('>I', rec_header[4:8])[0] * 2
            content = f.read(content_len)
            if len(content) < 4:
                records.append(NULL.copy()); continue
            stype = struct.unpack('<I', content[0:4])[0]
            if stype == 0 or len(content) < 44:
                records.append(NULL.copy()); continue
            bbox       = struct.unpack('<4d', content[4:36])
            num_parts  = struct.unpack('<I', content[36:40])[0]
            num_points = struct.unpack('<I', content[40:44])[0]
            part_starts = [struct.unpack('<I', content[44 + i*4: 48 + i*4])[0]
                           for i in range(num_parts)]
            part_starts.append(num_points)
            pts_offset = 44 + num_parts * 4
            if pts_offset + num_points * 16 > len(content):
                records.append(NULL.copy()); continue
            all_xy = struct.unpack(f'<{num_points*2}d',
                                   content[pts_offset: pts_offset + num_points * 16])
            xs = all_xy[0::2]
            ys = all_xy[1::2]

            if is_mercator:
                coords = [merc_to_wgs84(x, y) for x, y in zip(xs, ys)]
                lon0, lat0 = merc_to_wgs84(bbox[0], bbox[1])
                lon1, lat1 = merc_to_wgs84(bbox[2], bbox[3])
                out_bbox = (lon0, lat0, lon1, lat1)
            else:
                coords = list(zip(xs, ys))
                out_bbox = bbox

            parts = []
            for i in range(num_parts):
                s, e = part_starts[i], part_starts[i + 1]
                parts.append(coords[s:e])

            records.append({'bbox': out_bbox, 'parts': parts})
    return records

def read_shp_wgs84(shp_path):
    """Read WGS84 polygon shapefile (block groups from Census)."""
    records = []
    with open(shp_path, 'rb') as f:
        f.read(100)
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            content_len = struct.unpack('>I', rec_header[4:8])[0] * 2
            content = f.read(content_len)
            if len(content) < 44:
                break
            stype = struct.unpack('<I', content[0:4])[0]
            if stype == 0:
                records.append({'bbox': None, 'parts': []})
                continue
            bbox       = struct.unpack('<4d', content[4:36])
            num_parts  = struct.unpack('<I', content[36:40])[0]
            num_points = struct.unpack('<I', content[40:44])[0]
            part_starts = [struct.unpack('<I', content[44 + i*4: 48 + i*4])[0]
                           for i in range(num_parts)]
            part_starts.append(num_points)
            pts_offset = 44 + num_parts * 4
            all_xy = struct.unpack(f'<{num_points*2}d',
                                   content[pts_offset: pts_offset + num_points * 16])
            xs = all_xy[0::2]
            ys = all_xy[1::2]
            coords = list(zip(xs, ys))
            parts = []
            for i in range(num_parts):
                s, e = part_starts[i], part_starts[i + 1]
                parts.append(coords[s:e])
            records.append({'bbox': bbox, 'parts': parts})
    return records

# ─────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
trips     = pd.read_csv('processed_data/tampatrips_1.csv')
zon_csv   = pd.read_csv('processed_data/tpa_zon_geoid.csv').dropna(subset=['ZONECLASS'])
zone_code = pd.read_csv('processed_data/zone_features_combined.csv')

trips['active_travel'] = trips['mode_type'].isin([1, 2]).astype(int)

# Load embeddings for new maps
try:
    zone_emb  = pd.read_csv('zone_embeddings.csv')
    emb_cols  = [c for c in zone_emb.columns if c.startswith('emb_')]
    HAS_EMBEDDINGS = True
    print(f"Embeddings loaded: {len(zone_emb)} zones × {len(emb_cols)} dims")
except FileNotFoundError:
    HAS_EMBEDDINGS = False
    print("zone_embeddings.csv not found — skipping embedding maps")

# Block-group aggregated features
def zone_entropy_fn(group):
    areas = group['ShapeSTAre']
    props = areas / areas.sum()
    return -(props[props > 0] * np.log(props[props > 0])).sum()

def weighted_mean(group, col):
    mask = group[col].notna()
    if mask.sum() == 0: return np.nan
    return np.average(group.loc[mask, col], weights=group.loc[mask, 'ShapeSTAre'])

dominant = (zon_csv.groupby('GEOID')
            .apply(lambda g: g.loc[g['ShapeSTAre'].idxmax(), 'ZONECLASS'])
            .reset_index().rename(columns={0: 'dominant_zone'}))
entropy  = (zon_csv.groupby('GEOID')
            .apply(zone_entropy_fn)
            .reset_index().rename(columns={0: 'zoning_entropy'}))

zon_cm = zon_csv.merge(zone_code[['zone_class','ped_score_norm']],
                        left_on='ZONECLASS', right_on='zone_class', how='left')
ped_score = (zon_cm.groupby('GEOID')
             .apply(lambda g: weighted_mean(g, 'ped_score_norm'))
             .reset_index().rename(columns={0: 'ped_score_weighted'}))

at_rate = (trips.groupby('o_bg')['active_travel']
           .mean().reset_index()
           .rename(columns={'o_bg':'GEOID','active_travel':'at_rate'}))

bg_stats = (dominant.merge(entropy,   on='GEOID')
                     .merge(ped_score, on='GEOID')
                     .merge(at_rate,   on='GEOID', how='left'))
bg_stats['zone_context'] = 'Other/PD'
bg_stats.loc[bg_stats['dominant_zone'].isin(RESIDENTIAL),     'zone_context'] = 'Residential'
bg_stats.loc[bg_stats['dominant_zone'].isin(NON_RESIDENTIAL), 'zone_context'] = 'Non-Residential'

# Build embedding-based block-group features (if embeddings available)
if HAS_EMBEDDINGS:
    emb_lookup = zone_emb.set_index('zone_class')[emb_cols]
    dom_emb_df = dominant.merge(emb_lookup, left_on='dominant_zone',
                                 right_index=True, how='left')
    emb_matrix = dom_emb_df[emb_cols].fillna(0).values

    # PCA to 2D: reduces 32-dim embeddings to mappable coordinates while
    # preserving the main axis of regulatory variation across zone classes.
    pca_map = PCA(n_components=2, random_state=42)
    pca_coords = pca_map.fit_transform(emb_matrix)
    dom_emb_df['emb_pc1'] = pca_coords[:, 0]
    dom_emb_df['emb_pc2'] = pca_coords[:, 1]

    # k=5 chosen to match Tampa's major regulatory families (RS, RM, commercial,
    # special districts, institutional). Elbow plot confirmed diminishing returns beyond 5.
    CLUSTER_LABELS = {
        0: 'Base Residential',
        1: 'Planned Development',
        2: 'Commercial / Dense Res.',
        3: 'Special Districts',
        4: 'Institutional / Airport',
    }
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    dom_emb_df['emb_cluster'] = km.fit_predict(emb_matrix)

    bg_stats = bg_stats.merge(
        dom_emb_df[['GEOID','emb_pc1','emb_pc2','emb_cluster']],
        on='GEOID', how='left'
    )
    print("Embedding PCA and clusters added to block groups")

# Centroid fallback from CSV
centroids = zon_csv.groupby('GEOID').agg(
    lat=('INTPTLAT','first'), lon=('INTPTLON','first')
).reset_index()
bg_stats = bg_stats.merge(centroids, on='GEOID', how='left')

print(f"Block groups: {len(bg_stats)}")

# ─────────────────────────────────────────────────────────────
# 3. READ ZONING SHAPEFILE
# ─────────────────────────────────────────────────────────────
print("Reading zoning shapefile...")
zon_dbf  = read_dbf('raw_data/shapefiles/Zoning/Zoning_District.dbf')
zon_shp  = read_shp_polygons('raw_data/shapefiles/Zoning/Zoning_District.shp', is_mercator=True)
assert len(zon_dbf) == len(zon_shp), "DBF/SHP record count mismatch"

for i, (dbf_row, shp_rec) in enumerate(zip(zon_dbf, zon_shp)):
    shp_rec['ZONECLASS'] = dbf_row.get('ZONECLASS', '')
    shp_rec['ZONEDESC']  = dbf_row.get('ZONEDESC', '')
    # Tag records with invalid area for later filtering
    area_str = dbf_row.get('ShapeSTAre', '').replace('*', '').strip()
    try:
        shp_rec['area_valid'] = float(area_str) > 0
    except:
        shp_rec['area_valid'] = False

# Bounding box for Tampa MSA (filter out any stray geometries)
LON_MIN, LON_MAX = -82.70, -82.15
LAT_MIN, LAT_MAX =  27.80,  28.20

def in_bounds(rec):
    if rec['bbox'] is None: return False
    lon0, lat0, lon1, lat1 = rec['bbox']
    return (lon0 > LON_MIN and lon1 < LON_MAX and
            lat0 > LAT_MIN and lat1 < LAT_MAX)

zon_shp = [r for r in zon_shp
           if in_bounds(r)
           and r.get('area_valid', True)
           and r.get('ZONECLASS', '').strip() != '']
print(f"Zoning polygons in bounds (land area > 0): {len(zon_shp)}")

# ─────────────────────────────────────────────────────────────
# 4. READ BLOCK GROUP SHAPEFILE (if available)
# ─────────────────────────────────────────────────────────────
BG_SHP_PATH = 'raw_data/tl_2025_12_bg/tl_2025_12_bg.shp'
HAS_BG_SHP  = os.path.exists(BG_SHP_PATH)

if HAS_BG_SHP:
    print("Reading block group shapefile...")
    bg_dbf = read_dbf('raw_data/tl_2025_12_bg/tl_2025_12_bg.dbf')
    bg_shp = read_shp_wgs84(BG_SHP_PATH)
    # Build GEOID lookup: only Hillsborough County (FIPS 12057)
    bg_lookup = {}
    for dbf_row, shp_rec in zip(bg_dbf, bg_shp):
        geoid_str = dbf_row.get('GEOID','')
        try:
            geoid_int = int(geoid_str)
        except:
            continue
        # Filter: Hillsborough County only, has geometry, ALAND > 0
        aland = dbf_row.get('ALAND','0').strip()
        try:
            land_ok = int(aland) > 0
        except:
            land_ok = False
        if geoid_str.startswith('12057') and shp_rec['parts'] and land_ok:
            bg_lookup[geoid_int] = shp_rec
    print(f"Hillsborough County block groups loaded (ALAND > 0): {len(bg_lookup)}")
else:
    print("tl_2025_12_bg.shp not found — using centroid scatter for BG maps")
    print("Drop tl_2025_12_bg.shp in this folder and rerun for polygon BG maps")

# ─────────────────────────────────────────────────────────────
# 5. DRAWING HELPERS
# ─────────────────────────────────────────────────────────────
def setup_ax(ax, title, subtitle=None):
    ax.set_xlim(LON_MIN + 0.05, LON_MAX + 0.05)
    ax.set_ylim(LAT_MIN + 0.02, LAT_MAX + 0.02)
    ax.set_facecolor(LAND_COLOR)    # uniform land background
    # Land background rectangle behind all polygons
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((LON_MIN, LAT_MIN), LON_MAX - LON_MIN,
                            LAT_MAX - LAT_MIN, color=LAND_COLOR, zorder=0))
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=9, fontweight='bold')

def draw_zone_polygons(ax, shp_records, color_fn, alpha=0.85, lw=0.05):
    """Draw zoning polygons; color_fn(record) -> color string."""
    patches = []
    colors  = []
    for rec in shp_records:
        c = color_fn(rec)
        for part in rec['parts']:
            if len(part) < 3:
                continue
            poly = MplPolygon(part, closed=True)
            patches.append(poly)
            colors.append(c)
    pc = PatchCollection(patches, facecolor=colors, edgecolor='none',
                         linewidth=lw, alpha=alpha)
    ax.add_collection(pc)

def draw_bg_polygons(ax, bg_stats_df, value_col, cmap_name,
                     vmin=None, vmax=None, label=''):
    """Draw block group polygons coloured by a continuous value."""
    col_data = bg_stats_df.set_index('GEOID')[value_col]
    if vmin is None: vmin = col_data.dropna().quantile(0.02)
    if vmax is None: vmax = col_data.dropna().quantile(0.98)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap   = cm.get_cmap(cmap_name)
    patches = []
    face_colors = []
    for geoid, row in bg_stats_df.set_index('GEOID').iterrows():
        shp_rec = bg_lookup.get(geoid)
        if shp_rec is None:
            continue
        val = row.get(value_col)
        color = cmap(norm(val)) if pd.notna(val) else '#CCCCCC'
        for part in shp_rec['parts']:
            if len(part) < 3: continue
            patches.append(MplPolygon(part, closed=True))
            face_colors.append(color)
    pc = PatchCollection(patches, facecolor=face_colors,
                         edgecolor='#888888', linewidth=0.15, alpha=0.9)
    ax.add_collection(pc)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm

def draw_all_bg_outlines(ax):
    """Draw all Hillsborough BGs as a light land background with subtle borders."""
    patches = []
    for geoid, shp_rec in bg_lookup.items():
        for part in shp_rec['parts']:
            if len(part) < 3: continue
            patches.append(MplPolygon(part, closed=True))
    pc = PatchCollection(patches, facecolor=LAND_COLOR, edgecolor='#C8C0B0',
                         linewidth=0.25, alpha=1.0, zorder=1)
    ax.add_collection(pc)

def draw_bg_scatter(ax, bg_stats_df, value_col, cmap_name,
                    vmin=None, vmax=None):
    """Centroid scatter fallback for block group variable."""
    col_data = bg_stats_df[value_col].dropna()
    if vmin is None: vmin = col_data.quantile(0.02)
    if vmax is None: vmax = col_data.quantile(0.98)
    sc = ax.scatter(bg_stats_df['lon'], bg_stats_df['lat'],
                    c=bg_stats_df[value_col], cmap=cmap_name,
                    vmin=vmin, vmax=vmax,
                    s=18, alpha=0.85, zorder=4, linewidths=0)
    return sc

def draw_bg_categorical(ax, bg_stats_df, cluster_col, cluster_colors,
                         cluster_labels):
    """Draw block groups coloured by a categorical cluster variable."""
    if HAS_BG_SHP:
        for geoid, row in bg_stats_df.set_index('GEOID').iterrows():
            shp_rec = bg_lookup.get(geoid)
            if shp_rec is None: continue
            cluster = row.get(cluster_col)
            if pd.isna(cluster): continue
            color = cluster_colors.get(int(cluster), '#CCCCCC')
            patches = [MplPolygon(part, closed=True)
                       for part in shp_rec['parts'] if len(part) >= 3]
            if patches:
                pc = PatchCollection(patches, facecolor=color,
                                     edgecolor='#888888', linewidth=0.15,
                                     alpha=0.9)
                ax.add_collection(pc)
    else:
        for _, row in bg_stats_df.iterrows():
            cluster = row.get(cluster_col)
            if pd.isna(cluster): continue
            color = cluster_colors.get(int(cluster), '#CCCCCC')
            ax.scatter(row['lon'], row['lat'], c=color, s=18,
                       alpha=0.85, zorder=4, linewidths=0)
    # Legend
    legend_patches = [
        mpatches.Patch(color=cluster_colors.get(k, '#CCC'),
                       label=cluster_labels.get(k, f'Cluster {k}'))
        for k in sorted(cluster_colors.keys())
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc='upper right',
              framealpha=0.9)

# ─────────────────────────────────────────────────────────────
# 6. MAP 1 — ZONE CONTEXT (Residential / Non-Res / Other)
# ─────────────────────────────────────────────────────────────
print("\nRendering Map 1: Zone Context...")
fig1, ax1 = plt.subplots(figsize=(9, 8))
setup_ax(ax1, "Zoning Context — Tampa MSA",
         "Residential | Non-Residential | Other/Planned Development")

ctx_color_fn = lambda rec: (
    RES_COLOR if rec['ZONECLASS'] in RESIDENTIAL else
    NRE_COLOR if rec['ZONECLASS'] in NON_RESIDENTIAL else
    OTH_COLOR
)
if HAS_BG_SHP:
    draw_all_bg_outlines(ax1)
draw_zone_polygons(ax1, zon_shp, ctx_color_fn, alpha=0.88)

legend_patches = [
    mpatches.Patch(color=RES_COLOR, label='Residential'),
    mpatches.Patch(color=NRE_COLOR, label='Non-Residential'),
    mpatches.Patch(color=OTH_COLOR, label='Other / Planned Development'),
]
ax1.legend(handles=legend_patches, loc='upper right', fontsize=8,
           framealpha=0.9)
plt.tight_layout()
plt.savefig("map1_zone_context.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: map1_zone_context.png")

# ─────────────────────────────────────────────────────────────
# 7. MAP 2 — TOP ZONE CLASSES (individual colours)
# ─────────────────────────────────────────────────────────────
print("Rendering Map 2: Zone Classes...")
# Keep top 15 zone classes by area; everything else → OTHER
top_zones = set(
    pd.Series([r['ZONECLASS'] for r in zon_shp])
    .value_counts().head(15).index
)
zone_color_fn = lambda rec: ZONE_PALETTE.get(rec['ZONECLASS'],
                              ZONE_PALETTE.get('OTHER', '#EEEEEE'))

fig2, ax2 = plt.subplots(figsize=(9, 8))
setup_ax(ax2, "Zone Class Distribution — Tampa MSA", "Top zone classes by area")
if HAS_BG_SHP:
    draw_all_bg_outlines(ax2)
draw_zone_polygons(ax2, zon_shp, zone_color_fn, alpha=0.85)

legend_patches2 = [
    mpatches.Patch(color=ZONE_PALETTE.get(z, '#EEEEEE'), label=z)
    for z in list(ZONE_PALETTE.keys())[:16]
    if z != 'OTHER'
]
legend_patches2.append(mpatches.Patch(color='#EEEEEE', label='Other'))
ax2.legend(handles=legend_patches2, loc='upper right', fontsize=6,
           ncol=2, framealpha=0.9)
plt.tight_layout()
plt.savefig("map2_zone_classes.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: map2_zone_classes.png")

# ─────────────────────────────────────────────────────────────
# 8. MAPS 3-5 — BLOCK GROUP CHOROPLETHS
# ─────────────────────────────────────────────────────────────
bg_map_specs = [
    ('zoning_entropy',    'plasma', 'map3_bg_entropy.png',
     'Zoning Entropy by Block Group',
     'Shannon entropy of zone-class area shares (higher = more regulatory diversity)',
     'Zoning Entropy'),
    ('ped_score_weighted','plasma', 'map4_bg_ped_score.png',
     'Pedestrian Orientation Score by Block Group',
     'Area-weighted code-derived pedestrian score (0–1)',
     'Ped. Score (0–1)'),
    ('at_rate',           'plasma',  'map5_bg_at_rate.png',
     'Active Travel Rate by Origin Block Group',
     'Share of trips originating in each block group that are walk or bike',
     'AT Rate'),
]

for value_col, cmap, fname, title, subtitle, cbar_label in bg_map_specs:
    print(f"Rendering {fname}...")
    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, title, subtitle)

    # Draw zoning as a light base layer for geographic context
    draw_zone_polygons(ax, zon_shp, lambda r: '#E8EAF6',
                       alpha=0.4, lw=0.0)

    if HAS_BG_SHP:
        # Draw all BGs as land background first
        draw_all_bg_outlines(ax)
        sm = draw_bg_polygons(ax, bg_stats, value_col, cmap)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    else:
        sc = draw_bg_scatter(ax, bg_stats, value_col, cmap)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        ax.text(0.01, 0.01,
                "Note: using centroids — add tl_2025_12_bg.shp for polygon map",
                transform=ax.transAxes, fontsize=6, color='gray',
                va='bottom')

    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# ─────────────────────────────────────────────────────────────
# 9. MAPS 7-9 — EMBEDDING-BASED MAPS
# ─────────────────────────────────────────────────────────────
if HAS_EMBEDDINGS and 'emb_pc1' in bg_stats.columns:

    # Map 7 — PC1 of dominant zone embedding
    print("Rendering Map 7: Embedding PC1...")
    fig7, ax7 = plt.subplots(figsize=(9, 8))
    setup_ax(ax7, "Regulatory Character — Embedding PC1",
             f"PC1 of dominant zone embedding ({pca_map.explained_variance_ratio_[0]:.0%} variance)\n"
             "Low = base residential  |  High = commercial / special districts")
    draw_zone_polygons(ax7, zon_shp, lambda r: '#E8EAF6', alpha=0.3, lw=0.0)
    if HAS_BG_SHP:
        draw_all_bg_outlines(ax7)
        sm7 = draw_bg_polygons(ax7, bg_stats, 'emb_pc1', 'plasma')
        cbar7 = plt.colorbar(sm7, ax=ax7, shrink=0.6, pad=0.02)
    else:
        sc7 = draw_bg_scatter(ax7, bg_stats, 'emb_pc1', 'plasma')
        cbar7 = plt.colorbar(sc7, ax=ax7, shrink=0.6, pad=0.02)
    cbar7.set_label('PC1 Score', fontsize=8)
    cbar7.ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig("map7_bg_emb_pc1.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: map7_bg_emb_pc1.png")

    # Map 8 — Embedding cluster (regulatory character)
    print("Rendering Map 8: Embedding Clusters...")
    # Cluster colors chosen to be maximally distinguishable: each major regulatory
    # family gets a unique hue so clusters are readable in the combined panel.
    CLUSTER_COLORS = {
        0: '#1976D2',   # Base Residential — blue
        1: '#9C27B0',   # Planned Development — purple
        2: '#E65100',   # Commercial / Dense — orange
        3: '#00838F',   # Special Districts — teal
        4: '#B71C1C',   # Institutional / Airport — red
    }
    fig8, ax8 = plt.subplots(figsize=(9, 8))
    setup_ax(ax8, "Regulatory Character Clusters",
             "K-means clustering (k=5) of dominant zone code embeddings")
    draw_zone_polygons(ax8, zon_shp, lambda r: '#E8EAF6', alpha=0.3, lw=0.0)
    if HAS_BG_SHP:
        draw_all_bg_outlines(ax8)
    draw_bg_categorical(ax8, bg_stats, 'emb_cluster',
                        CLUSTER_COLORS, CLUSTER_LABELS)
    plt.tight_layout()
    plt.savefig("map8_bg_emb_clusters.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: map8_bg_emb_clusters.png")

    # Map 9 — Ped score (fill) vs AT rate (contour overlay)
    print("Rendering Map 9: Ped Score vs. AT Rate...")
    fig9, axes9 = plt.subplots(1, 2, figsize=(16, 8))
    fig9.suptitle("Pedestrian Orientation Score vs. Active Travel Rate",
                  fontsize=12, fontweight='bold')

    for ax9, col, cmap, title9, lbl9 in [
        (axes9[0], 'ped_score_weighted', 'plasma',
         'Ped. Orientation Score\n(from zoning code rubric)', 'Ped. Score'),
        (axes9[1], 'at_rate', 'plasma',
         'Active Travel Rate\n(share of walk/bike trips)', 'AT Rate'),
    ]:
        setup_ax(ax9, title9)
        draw_zone_polygons(ax9, zon_shp, lambda r: '#E8EAF6', alpha=0.3, lw=0.0)
        if HAS_BG_SHP:
            draw_all_bg_outlines(ax9)
            sm9 = draw_bg_polygons(ax9, bg_stats, col, cmap)
            cbar9 = plt.colorbar(sm9, ax=ax9, shrink=0.6, pad=0.02)
        else:
            sc9 = draw_bg_scatter(ax9, bg_stats, col, cmap)
            cbar9 = plt.colorbar(sc9, ax=ax9, shrink=0.6, pad=0.02)
        cbar9.set_label(lbl9, fontsize=8)
        cbar9.ax.tick_params(labelsize=7)

    # Correlation annotation
    valid = bg_stats[['ped_score_weighted','at_rate']].dropna()
    if len(valid) > 10:
        corr = valid['ped_score_weighted'].corr(valid['at_rate'])
        axes9[1].text(0.02, 0.02,
                      f"Pearson r = {corr:.3f}\n(ped score vs AT rate)",
                      transform=axes9[1].transAxes, fontsize=8,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='#BDBDBD'),
                      va='bottom')

    plt.tight_layout()
    plt.savefig("map9_ped_vs_at.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: map9_ped_vs_at.png")

else:
    print("Skipping embedding maps (zone_embeddings.csv not found)")
    
# ─────────────────────────────────────────────────────────────
# VIZ 8 — ZONE CONTEXT + ACTIVE TRAVEL RATE (side-by-side)
#   Left:  Block groups coloured by dominant zone context
#   Right: Block groups coloured by active travel rate
#   Purpose: orient the audience to Tampa's spatial structure
#            before showing DML results
# ─────────────────────────────────────────────────────────────
print("\nRendering Viz 8: Zone Context + AT Rate (presentation map)...")

fig_v8, (ax_ctx, ax_at) = plt.subplots(1, 2, figsize=(18, 8))
fig_v8.suptitle("Tampa MSA — Zoning Context and Active Travel Geography",
                fontsize=14, fontweight='bold', y=0.98)

# ── Shared: draw zoning polygons as faint base layer on both panels ──
for ax in [ax_ctx, ax_at]:
    setup_ax(ax, "")
    draw_zone_polygons(ax, zon_shp, lambda r: '#E8EAF6', alpha=0.25, lw=0.0)

# ══════════════════════════════════════════════════════════════
# LEFT PANEL: Block groups by dominant zone context
# ══════════════════════════════════════════════════════════════
ax_ctx.set_title("Dominant Zone Context by Block Group",
                 fontsize=11, fontweight='bold')

CTX_COLORS = {
    'Residential':     '#66BB6A',
    'Non-Residential': '#FFA726',
    'Other/PD':        '#BDBDBD',
}

if HAS_BG_SHP:
    draw_all_bg_outlines(ax_ctx)
    # Draw each BG coloured by its zone context
    for geoid, row in bg_stats.set_index('GEOID').iterrows():
        shp_rec = bg_lookup.get(geoid)
        if shp_rec is None:
            continue
        ctx = row.get('zone_context', 'Other/PD')
        color = CTX_COLORS.get(ctx, '#CCCCCC')
        patches_list = [MplPolygon(part, closed=True)
                        for part in shp_rec['parts'] if len(part) >= 3]
        if patches_list:
            pc = PatchCollection(patches_list, facecolor=color,
                                 edgecolor='#888888', linewidth=0.2,
                                 alpha=0.85, zorder=2)
            ax_ctx.add_collection(pc)
else:
    # Centroid scatter fallback
    for ctx, color in CTX_COLORS.items():
        mask = bg_stats['zone_context'] == ctx
        subset = bg_stats[mask]
        ax_ctx.scatter(subset['lon'], subset['lat'],
                       c=color, s=25, alpha=0.8, zorder=4,
                       edgecolors='white', linewidths=0.3)

# Legend
legend_patches_ctx = [
    mpatches.Patch(color=c, label=l) for l, c in CTX_COLORS.items()
]
ax_ctx.legend(handles=legend_patches_ctx, loc='upper right', fontsize=9,
              framealpha=0.9)



# ══════════════════════════════════════════════════════════════
# RIGHT PANEL: Block groups by active travel rate
# ══════════════════════════════════════════════════════════════
ax_at.set_title("Active Travel Rate by Origin Block Group",
                fontsize=11, fontweight='bold')

# Custom diverging colormap anchored at cream midpoint: terracotta (low AT)
# to teal (high AT) matches the paper's palette and avoids the perceptual
# issues of default diverging cmaps on non-zero-centered data.
from matplotlib.colors import LinearSegmentedColormap
at_cmap = LinearSegmentedColormap.from_list(
    'at_div', ['#C75B39', '#F5F0E8', '#1B8A6B'], N=256
)

at_valid = bg_stats['at_rate'].dropna()
at_vmin, at_vmax = 0.0, min(at_valid.quantile(0.95), 0.70)

if HAS_BG_SHP:
    draw_all_bg_outlines(ax_at)
    # Draw each BG coloured by AT rate
    norm_at = mcolors.Normalize(vmin=at_vmin, vmax=at_vmax)
    patches_at = []
    fcolors_at = []
    for geoid, row in bg_stats.set_index('GEOID').iterrows():
        shp_rec = bg_lookup.get(geoid)
        if shp_rec is None:
            continue
        val = row.get('at_rate')
        if pd.notna(val):
            color = at_cmap(norm_at(val))
        else:
            color = '#CCCCCC'
        for part in shp_rec['parts']:
            if len(part) < 3:
                continue
            patches_at.append(MplPolygon(part, closed=True))
            fcolors_at.append(color)
    pc_at = PatchCollection(patches_at, facecolor=fcolors_at,
                            edgecolor='#888888', linewidth=0.2,
                            alpha=0.9, zorder=2)
    ax_at.add_collection(pc_at)
    sm_at = cm.ScalarMappable(cmap=at_cmap, norm=norm_at)
    sm_at.set_array([])
    cbar_at = plt.colorbar(sm_at, ax=ax_at, shrink=0.6, pad=0.02)
else:
    # Centroid scatter fallback
    sc_at = ax_at.scatter(
        bg_stats['lon'], bg_stats['lat'],
        c=bg_stats['at_rate'], cmap=at_cmap,
        vmin=at_vmin, vmax=at_vmax,
        s=25, alpha=0.85, zorder=4, linewidths=0
    )
    cbar_at = plt.colorbar(sc_at, ax=ax_at, shrink=0.6, pad=0.02)
    ax_at.text(0.01, 0.01,
               "Note: using centroids — add tl_2025_12_bg.shp for polygon map",
               transform=ax_at.transAxes, fontsize=6, color='gray', va='bottom')

cbar_at.set_label('Active Travel Rate (walk + bike share)', fontsize=9)
cbar_at.ax.tick_params(labelsize=8)

# Summary annotation box
n_bg_with_at = bg_stats['at_rate'].notna().sum()
mean_at = bg_stats['at_rate'].mean()
median_at = bg_stats['at_rate'].median()
summary_text = (
    f"n = {len(bg_stats)} block groups\n"
    f"{n_bg_with_at} with trip data\n"
    f"Mean AT rate: {mean_at:.1%}\n"
    f"Median AT rate: {median_at:.1%}"
)
ax_at.text(0.02, 0.02, summary_text,
           transform=ax_at.transAxes, fontsize=8, va='bottom',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#BDBDBD', alpha=0.9),
           zorder=5)

plt.tight_layout()
plt.savefig("viz8_spatial_context_at.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: viz8_spatial_context_at.png")

# ─────────────────────────────────────────────────────────────
# 9. MAP 6 — COMBINED 2×3 PANEL
# ─────────────────────────────────────────────────────────────
print("Rendering Map 6: Combined panel...")
fig6, axes = plt.subplots(2, 4, figsize=(26, 14))
fig6.suptitle("Tampa MSA — Zoning and Active Travel Spatial Overview",
              fontsize=14, fontweight='bold', y=1.01)

panel_specs = [
    (axes[0][0], "Zone Context",              'context'),
    (axes[0][1], "Zone Classes",              'classes'),
    (axes[0][2], "Zoning Entropy",            'entropy'),
    (axes[0][3], "Ped. Orientation Score",    'ped'),
    (axes[1][0], "Active Travel Rate",        'at_rate'),
    (axes[1][1], "Regulatory Character PC1",  'emb_pc1'),
    (axes[1][2], "Regulatory Clusters",       'emb_cluster'),
    (axes[1][3], "",                          'blank'),
]

for ax, title, ptype in panel_specs:
    if ptype == 'blank':
        ax.axis('off')
        # Add summary text
        summary = (
            "Data Summary\n\n"
            f"Trips: 5,749\n"
            f"Block groups: {len(bg_stats)}\n"
            f"Active travel rate: 26.1%\n"
            f"Median trip distance: 1.38 mi\n\n"
            f"Zoning parcels: 3,819\n"
            f"Zone classes: 55\n"
            f"Dominant zone: RS-60 (23.7% by area)\n\n"
            f"Block group stats\n"
            f"Entropy mean: {bg_stats['zoning_entropy'].mean():.2f}\n"
            f"Ped. score mean: {bg_stats['ped_score_weighted'].mean():.2f}\n"
            f"AT rate median: {bg_stats['at_rate'].median()*100:.1f}%"
        )
        ax.text(0.5, 0.5, summary, transform=ax.transAxes,
                fontsize=9, va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5',
                          edgecolor='#BDBDBD', linewidth=1))
        continue

    setup_ax(ax, title)
    draw_zone_polygons(ax, zon_shp, lambda r: '#E8EAF6', alpha=0.3, lw=0.0)

    if ptype == 'context':
        if HAS_BG_SHP:
            draw_all_bg_outlines(ax)
        draw_zone_polygons(ax, zon_shp, ctx_color_fn, alpha=0.85)
        for label, color in [('Residential',RES_COLOR),
                              ('Non-Res.',NRE_COLOR),('Other',OTH_COLOR)]:
            ax.plot([], [], color=color, linewidth=6,
                    label=label, solid_capstyle='butt')
        ax.legend(fontsize=6, loc='upper right', framealpha=0.85)

    elif ptype == 'classes':
        if HAS_BG_SHP:
            draw_all_bg_outlines(ax)
        draw_zone_polygons(ax, zon_shp, zone_color_fn, alpha=0.85)

    elif ptype in ('entropy','ped','at_rate','emb_pc1'):
        col = {'entropy':'zoning_entropy', 'ped':'ped_score_weighted',
               'at_rate':'at_rate', 'emb_pc1':'emb_pc1'}[ptype]
        lbl = {'entropy':'Entropy','ped':'Score',
               'at_rate':'AT Rate','emb_pc1':'PC1'}[ptype]

        if HAS_BG_SHP:
            draw_all_bg_outlines(ax)
            sm = draw_bg_polygons(ax, bg_stats, col, 'plasma')
            cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
        else:
            sc = draw_bg_scatter(ax, bg_stats, col, 'plasma')
            cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label(lbl, fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    elif ptype == 'emb_cluster':
        if HAS_EMBEDDINGS and 'emb_cluster' in bg_stats.columns:
            if HAS_BG_SHP:
                draw_all_bg_outlines(ax)
            draw_bg_categorical(ax, bg_stats, 'emb_cluster',
                                CLUSTER_COLORS, CLUSTER_LABELS)
        else:
            ax.text(0.5, 0.5, 'Embeddings\nnot available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=9, color='gray')

plt.tight_layout()
plt.savefig("map6_combined.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: map6_combined.png")

print("\n✓ All maps complete.")
if not HAS_BG_SHP:
    print("  To upgrade block group maps from centroids to polygons:")
    print("  Drop tl_2025_12_bg.shp in this folder and rerun the script.")

#the final file names are df (origin and destination in the county area),destination county, and origin county. df is the only one with geometry

# ============================================================
# Final MAP 1 & MAP 2 — Tampa zoning maps
# ============================================================

import os, struct, math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.collections import PatchCollection
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon

# ----------------------------
# Paths
# ----------------------------
ZONING_SHP = 'raw_data/shapefiles/Zoning/Zoning_District.shp'
ZONING_DBF = 'raw_data/shapefiles/Zoning/Zoning_District.dbf'

BG_SHP_PATH = 'raw_data/tl_2025_12_bg/tl_2025_12_bg.shp'
BG_DBF_PATH = 'raw_data/tl_2025_12_bg/tl_2025_12_bg.dbf'

CITY_BOUNDARY_SHP = 'raw_data/shapefiles/Zoning/Zoning_District.shp'

# ----------------------------
# Colors
# ----------------------------
RES_COLOR = '#66BB6A'
NRE_COLOR = '#FFA726'
OTH_COLOR = '#BDBDBD'

WATER_COLOR = '#EAF3FB'
LAND_COLOR = '#F7F7F7'

RESIDENTIAL = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO','RO-1','SH-RS','SH-RS-A','SH-RM','SH-RO','SH-PD',
    'YC-2','YC-4','YC-8','YC-9',
]

NON_RESIDENTIAL = [
    'CG','CI','CN','OP','OP-1','IG','IH','CBD-1','CBD-2',
    'CD-1','CD-2','CD-3','NMU-35','NMU-24','NMU-16',
    'SH-CG','SH-CI','SH-CN','YC-1','YC-3','YC-5','YC-6','YC-7',
]

ZONE_PALETTE = {
    'RS-60':  '#1565C0', 'RS-50':  '#1976D2', 'RS-75':  '#2196F3',
    'RS-100': '#42A5F5', 'RS-150': '#90CAF9',
    'PD':     '#9C27B0', 'PD-A':   '#CE93D8',
    'CG':     '#E65100', 'CI':     '#F57C00', 'CN':     '#FFA726',
    'RM-16':  '#2E7D32', 'RM-24':  '#43A047', 'RM-18':  '#66BB6A',
    'IH':     '#B71C1C', 'IG':     '#E53935',
    'SH-RS':  '#00838F', 'SH-CG':  '#FF8F00',
    'CBD-2':  '#4A148C', 'CBD-1':  '#6A1B9A',
    'OTHER':  '#EEEEEE',
}

# ----------------------------
# Shapefile readers
# ----------------------------
def merc_to_wgs84(x, y):
    lon = math.degrees(x / 6378137.0)
    lat = math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
    return lon, lat

def read_dbf(path):
    with open(path, 'rb') as f:
        header = f.read(32)
        n_records = struct.unpack('<I', header[4:8])[0]
        header_size = struct.unpack('<H', header[8:10])[0]
        record_size = struct.unpack('<H', header[10:12])[0]

        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D or len(fd) < 32:
                break
            name = fd[:11].replace(b'\x00', b'').decode('ascii', 'ignore').strip()
            ftype = chr(fd[11])
            flen = fd[16]
            fields.append((name, ftype, flen))

        f.seek(header_size)
        rows = []

        for _ in range(n_records):
            rec = f.read(record_size)
            if not rec or rec[0] == 0x1A:
                break

            row = {}
            pos = 1

            for name, ftype, flen in fields:
                val = rec[pos:pos + flen].decode('ascii', 'ignore').strip()
                row[name] = val
                pos += flen

            rows.append(row)

    return rows

def read_shp_polygons(shp_path, is_mercator=True):
    records = []

    with open(shp_path, 'rb') as f:
        f.read(100)

        while True:
            rec_header = f.read(8)

            if len(rec_header) < 8:
                break

            content_len = struct.unpack('>I', rec_header[4:8])[0] * 2
            content = f.read(content_len)

            if len(content) < 44:
                records.append({'bbox': None, 'parts': []})
                continue

            stype = struct.unpack('<I', content[0:4])[0]

            if stype == 0:
                records.append({'bbox': None, 'parts': []})
                continue

            bbox = struct.unpack('<4d', content[4:36])
            num_parts = struct.unpack('<I', content[36:40])[0]
            num_points = struct.unpack('<I', content[40:44])[0]

            part_starts = [
                struct.unpack('<I', content[44 + i * 4:48 + i * 4])[0]
                for i in range(num_parts)
            ]
            part_starts.append(num_points)

            pts_offset = 44 + num_parts * 4
            all_xy = struct.unpack(
                f'<{num_points * 2}d',
                content[pts_offset:pts_offset + num_points * 16]
            )

            xs = all_xy[0::2]
            ys = all_xy[1::2]

            if is_mercator:
                coords = [merc_to_wgs84(x, y) for x, y in zip(xs, ys)]
                lon0, lat0 = merc_to_wgs84(bbox[0], bbox[1])
                lon1, lat1 = merc_to_wgs84(bbox[2], bbox[3])
                out_bbox = (lon0, lat0, lon1, lat1)
            else:
                coords = list(zip(xs, ys))
                out_bbox = bbox

            parts = []

            for i in range(num_parts):
                s, e = part_starts[i], part_starts[i + 1]
                parts.append(coords[s:e])

            records.append({'bbox': out_bbox, 'parts': parts})

    return records

def read_shp_wgs84(shp_path):
    return read_shp_polygons(shp_path, is_mercator=False)

# ----------------------------
# Load zoning polygons
# ----------------------------
print("Reading zoning shapefile...")

zon_dbf = read_dbf(ZONING_DBF)
zon_shp = read_shp_polygons(ZONING_SHP, is_mercator=True)

for dbf_row, shp_rec in zip(zon_dbf, zon_shp):
    shp_rec['ZONECLASS'] = dbf_row.get('ZONECLASS', '')
    shp_rec['ZONEDESC'] = dbf_row.get('ZONEDESC', '')

    area_str = dbf_row.get('ShapeSTAre', '').replace('*', '').strip()

    try:
        shp_rec['area_valid'] = float(area_str) > 0
    except:
        shp_rec['area_valid'] = False

LON_MIN, LON_MAX = -82.70, -82.15
LAT_MIN, LAT_MAX = 27.80, 28.20

def in_bounds(rec):
    if rec['bbox'] is None:
        return False

    lon0, lat0, lon1, lat1 = rec['bbox']

    return (
        lon0 > LON_MIN and lon1 < LON_MAX and
        lat0 > LAT_MIN and lat1 < LAT_MAX
    )

zon_shp = [
    r for r in zon_shp
    if in_bounds(r)
    and r.get('area_valid', True)
    and r.get('ZONECLASS', '').strip() != ''
]

print(f"Zoning polygons loaded: {len(zon_shp)}")

# ----------------------------
# Map extent
# ----------------------------
def get_records_bbox(records, pad_ratio=0.08):
    xs, ys = [], []

    for rec in records:
        for part in rec['parts']:
            for x, y in part:
                xs.append(x)
                ys.append(y)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    xpad = (xmax - xmin) * pad_ratio
    ypad = (ymax - ymin) * pad_ratio

    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

MAP_XMIN, MAP_XMAX, MAP_YMIN, MAP_YMAX = get_records_bbox(zon_shp, pad_ratio=0.08)

# Expand slightly west/south to show Tampa Bay
MAP_XMIN -= 0.00
MAP_XMAX += 0.00
MAP_YMIN -= 0.00

# ----------------------------
# Load BG basemap
# ----------------------------
print("Reading block group basemap...")

TARGET_COUNTIES = {
    '12057': 'Hillsborough',
    '12103': 'Pinellas',
    '12101': 'Pasco',
    '12053': 'Hernando',
}

bg_lookup = {}

if os.path.exists(BG_SHP_PATH):
    bg_dbf = read_dbf(BG_DBF_PATH)
    bg_shp = read_shp_wgs84(BG_SHP_PATH)

    for dbf_row, shp_rec in zip(bg_dbf, bg_shp):
        geoid_str = dbf_row.get('GEOID', '')

        try:
            geoid_int = int(geoid_str)
        except:
            continue

        county_fips = geoid_str[:5]

        aland = dbf_row.get('ALAND', '0').strip()
        awater = dbf_row.get('AWATER', '0').strip()

        try:
            aland_val = int(aland)
            awater_val = int(awater)
        except:
            aland_val = 0
            awater_val = 0

        # Remove water-dominated BGs, especially Pinellas water polygons
        land_ok = (
            aland_val > 0 and
            aland_val / (aland_val + awater_val + 1) > 0.5
        )

        if county_fips in TARGET_COUNTIES and shp_rec['parts'] and land_ok:
            bg_lookup[geoid_int] = shp_rec

print(f"Block groups loaded for basemap: {len(bg_lookup)}")

# ----------------------------
# City boundary from TA_FLU
# ----------------------------
print("Reading and cleaning City of Tampa boundary...")

city_gdf = gpd.read_file(CITY_BOUNDARY_SHP)
city_gdf['geometry'] = city_gdf.geometry.buffer(0)

union_geom = unary_union(city_gdf.geometry)

polys = list(union_geom.geoms) if union_geom.geom_type == 'MultiPolygon' else [union_geom]

clean_polys = []

for p in polys:
    xmin, ymin, xmax, ymax = p.bounds

    # Remove far northeast tiny fragment only
    is_far_ne_tiny = (
        xmin > -9.165e6 and
        ymin > 3.267e6 and
        p.area < 2e7
    )

    if not is_far_ne_tiny:
        clean_polys.append(Polygon(p.exterior))

outer_geom = MultiPolygon(clean_polys)

city_boundary = gpd.GeoDataFrame(
    geometry=[outer_geom],
    crs=city_gdf.crs
)

city_boundary_wgs84 = city_boundary.to_crs(epsg=4326)

print("City boundary ready.")

# ----------------------------
# Drawing helpers
# ----------------------------
def setup_ax(ax, title, subtitle=None):
    ax.set_xlim(MAP_XMIN, MAP_XMAX)
    ax.set_ylim(MAP_YMIN, MAP_YMAX)

    ax.set_facecolor(WATER_COLOR)
    ax.set_aspect('equal')

    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')

    if subtitle:
        ax.set_title(
            f"{title}\n{subtitle}",
            fontsize=15,
            fontweight='semibold',
            pad=14
        )
    else:
        ax.set_title(
            title,
            fontsize=15,
            fontweight='semibold',
            pad=14
        )

def draw_all_bg_outlines(ax):
    patches = []

    for geoid, shp_rec in bg_lookup.items():
        for part in shp_rec['parts']:
            if len(part) < 3:
                continue
            patches.append(MplPolygon(part, closed=True))

    pc = PatchCollection(
        patches,
        facecolor=LAND_COLOR,
        edgecolor='#E2DDD7',
        linewidth=0.35,
        alpha=0.85,
        zorder=1
    )

    ax.add_collection(pc)

def draw_zone_polygons(ax, shp_records, color_fn, alpha=0.72, lw=0.04):
    patches = []
    colors = []

    for rec in shp_records:
        c = color_fn(rec)

        for part in rec['parts']:
            if len(part) < 3:
                continue

            patches.append(MplPolygon(part, closed=True))
            colors.append(c)

    pc = PatchCollection(
        patches,
        facecolor=colors,
        edgecolor='none',
        linewidth=lw,
        alpha=alpha,
        zorder=5
    )

    ax.add_collection(pc)

def add_north_arrow(ax):
    ax.annotate(
        'N',
        xy=(0.94, 0.91),
        xytext=(0.94, 0.83),
        xycoords='axes fraction',
        textcoords='axes fraction',
        ha='center',
        va='center',
        fontsize=13,
        fontweight='bold',
        color='#222222',
        arrowprops=dict(
            arrowstyle='-|>',
            lw=1.4,
            color='#222222',
            shrinkA=0,
            shrinkB=0
        ),
        zorder=50
    )

def add_scale_bar_miles(ax, length_miles=5):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    mean_lat = (y0 + y1) / 2
    miles_per_deg_lon = 69.172 * np.cos(np.deg2rad(mean_lat))
    length_deg = length_miles / miles_per_deg_lon

    x_margin = (x1 - x0) * 0.06
    y_margin = (y1 - y0) * 0.06

    lon_start = x1 - length_deg - x_margin
    lat_start = y0 + y_margin

    height = (y1 - y0) * 0.008

    ax.add_patch(
        Rectangle(
            (lon_start, lat_start),
            length_deg / 2,
            height,
            facecolor='black',
            edgecolor='black',
            lw=1.0,
            zorder=50
        )
    )

    ax.add_patch(
        Rectangle(
            (lon_start + length_deg / 2, lat_start),
            length_deg / 2,
            height,
            facecolor='white',
            edgecolor='black',
            lw=1.0,
            zorder=50
        )
    )

    ax.text(
        lon_start,
        lat_start - height * 2.4,
        '0',
        fontsize=12,
        ha='center',
        va='top',
        color='black',
        zorder=51
    )

    ax.text(
        lon_start + length_deg,
        lat_start - height * 2.4,
        '5 mi',
        fontsize=12,
        ha='center',
        va='top',
        color='black',
        zorder=51
    )

def add_tampa_bay_label(ax):
    ax.text(
        -82.58,
        27.89,
        'Tampa Bay',
        fontsize=16,
        fontstyle='italic',
        color='#4F7FA6',
        alpha=0.85,
        ha='center',
        va='center',
        zorder=60
    )

def draw_city_boundary(ax):
    city_boundary_wgs84.boundary.plot(
        ax=ax,
        color='black',
        linewidth=1.0,
        zorder=30
    )

# ============================================================
# MAP 1 — Zoning context
# ============================================================
print("Rendering Map 1...")

fig1, ax1 = plt.subplots(figsize=(12, 8.5))

setup_ax(
    ax1,
    "Zoning Context in the City of Tampa"
)

draw_all_bg_outlines(ax1)

ctx_color_fn = lambda rec: (
    RES_COLOR if rec['ZONECLASS'] in RESIDENTIAL else
    NRE_COLOR if rec['ZONECLASS'] in NON_RESIDENTIAL else
    OTH_COLOR
)

draw_zone_polygons(ax1, zon_shp, ctx_color_fn, alpha=0.72)
draw_city_boundary(ax1)
add_tampa_bay_label(ax1)
add_north_arrow(ax1)
add_scale_bar_miles(ax1, length_miles=5)

legend_patches1 = [
    mpatches.Patch(color=RES_COLOR, label='Residential'),
    mpatches.Patch(color=NRE_COLOR, label='Non-Residential'),
    mpatches.Patch(color=OTH_COLOR, label='Other / Planned Development'),
]

ax1.legend(
    handles=legend_patches1,
    loc='upper left',
    fontsize=12,
    #title='Zoning context',
    #title_fontsize=13,
    frameon=True,
    framealpha=0.95,
    edgecolor='#DDDDDD'
)

plt.tight_layout()
#plt.savefig("map1_zone_context_final.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig1)

print("Saved: map1_zone_context_final.png")

# ============================================================
# MAP 2 — Zone class distribution
# ============================================================
print("Rendering Map 2...")

fig2, ax2 = plt.subplots(figsize=(12, 8.5))

setup_ax(
    ax2,
    "Distribution of Major Zoning Classes in the City of Tampa"
)

draw_all_bg_outlines(ax2)

top_zones = set(
    pd.Series([r['ZONECLASS'] for r in zon_shp])
    .value_counts()
    .head(15)
    .index
)

def zone_color_fn(rec):
    z = rec['ZONECLASS']
    if z in top_zones:
        return ZONE_PALETTE.get(z, ZONE_PALETTE['OTHER'])
    return ZONE_PALETTE['OTHER']

draw_zone_polygons(ax2, zon_shp, zone_color_fn, alpha=0.72)
draw_city_boundary(ax2)
add_tampa_bay_label(ax2)
add_north_arrow(ax2)
add_scale_bar_miles(ax2, length_miles=5)

legend_patches2 = [
    mpatches.Patch(color=ZONE_PALETTE.get(z, '#EEEEEE'), label=z)
    for z in sorted(top_zones)
]

legend_patches2.append(
    mpatches.Patch(color=ZONE_PALETTE['OTHER'], label='Other')
)

ax2.legend(
    handles=legend_patches2,
    loc='upper left',
    fontsize=11,
    #title='Zone class',
    #title_fontsize=13,
    ncol=2,
    frameon=True,
    framealpha=0.95,
    edgecolor='#DDDDDD'
)

plt.tight_layout()
#plt.savefig("map2_zone_classes_final.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)

print("Saved: map2_zone_classes_final.png")

# ============================================================
# MAP 5 — Active Travel Rate (final version)
print("Rendering Map 5 (AT Rate)...")
fig5, ax5 = plt.subplots(figsize=(12, 8.5))
setup_ax(ax5, "Active Travel Rate by Origin Block Group")

# Background: all county BGs (gray fill for context)
draw_all_bg_outlines(ax5)

# Colored BG polygons on top
sm = draw_bg_polygons(ax5, bg_stats, 'at_rate', 'plasma')

# Black outline around study-area BGs on top of everything
study_geoids = set(bg_stats[bg_stats['at_rate'].notna()]['GEOID'].values)
outline_patches = []
for geoid, shp_rec in bg_lookup.items():
    if geoid in study_geoids or str(geoid) in study_geoids:
        for part in shp_rec['parts']:
            if len(part) < 3:
                continue
            outline_patches.append(MplPolygon(part, closed=True))
pc = PatchCollection(outline_patches, facecolor='none', edgecolor='black',
                     linewidth=0.8, zorder=29)
ax5.add_collection(pc)

add_tampa_bay_label(ax5)
add_north_arrow(ax5)
add_scale_bar_miles(ax5, length_miles=5)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cax = inset_axes(ax5, width="40%", height="3%", loc='upper left',
                 bbox_to_anchor=(0.02, -0.02, 1, 1), bbox_transform=ax5.transAxes)
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('AT Rate (walk + bike)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

n_bg_with_at = bg_stats['at_rate'].notna().sum()
mean_at = bg_stats['at_rate'].mean()
median_at = bg_stats['at_rate'].median()
ax5.text(0.01, 0.01,
         f"n = {n_bg_with_at} block groups\n"
         f"{n_bg_with_at} with trip data\n"
         f"Mean AT rate: {mean_at*100:.1f}%\n"
         f"Median AT rate: {median_at*100:.1f}%",
         transform=ax5.transAxes, fontsize=14,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DDDDDD', alpha=0.9))

plt.tight_layout()
plt.savefig("map5_bg_at_rate_final.png", dpi=300, bbox_inches="tight")
plt.close(fig5)
print("Saved: map5_bg_at_rate_final.png")
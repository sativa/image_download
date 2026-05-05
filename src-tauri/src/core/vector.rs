//! Vector file parser: GeoJSON / Shapefile / GPKG → bbox + GeoJSON geometry.

use anyhow::{anyhow, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ParsedVector {
    pub bbox: [f64; 4],
    pub geometry: geojson::Geometry,
    pub layer_count: u32,
}

pub fn parse_vector(path: &Path) -> Result<ParsedVector> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("geojson") | Some("json") => parse_geojson(path),
        Some("shp") => parse_shapefile(path),
        Some("gpkg") => parse_gpkg(path),
        Some("parquet") | Some("geoparquet") => parse_geoparquet(path),
        _ => Err(anyhow!("unsupported_format")),
    }
}

fn parse_geojson(path: &Path) -> Result<ParsedVector> {
    let s = std::fs::read_to_string(path)?;
    let gj: geojson::GeoJson = s.parse()?;
    let (geom, count) = match gj {
        geojson::GeoJson::FeatureCollection(fc) => {
            let n = fc.features.len() as u32;
            let g = fc
                .features
                .into_iter()
                .find_map(|f| f.geometry)
                .ok_or_else(|| anyhow!("no_geometry"))?;
            (g, n)
        }
        geojson::GeoJson::Feature(f) => (f.geometry.ok_or_else(|| anyhow!("no_geometry"))?, 1),
        geojson::GeoJson::Geometry(g) => (g, 1),
    };
    let bbox = geometry_bbox(&geom)?;
    Ok(ParsedVector {
        bbox,
        geometry: geom,
        layer_count: count,
    })
}

fn geometry_bbox(g: &geojson::Geometry) -> Result<[f64; 4]> {
    use geojson::Value;
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut feed = |x: f64, y: f64| {
        if x < min[0] {
            min[0] = x
        }
        if y < min[1] {
            min[1] = y
        }
        if x > max[0] {
            max[0] = x
        }
        if y > max[1] {
            max[1] = y
        }
    };
    match &g.value {
        Value::Point(c) => feed(c[0], c[1]),
        Value::LineString(cs) | Value::MultiPoint(cs) => {
            for c in cs {
                feed(c[0], c[1])
            }
        }
        Value::Polygon(rings) | Value::MultiLineString(rings) => {
            for ring in rings {
                for c in ring {
                    feed(c[0], c[1])
                }
            }
        }
        Value::MultiPolygon(polys) => {
            for poly in polys {
                for ring in poly {
                    for c in ring {
                        feed(c[0], c[1])
                    }
                }
            }
        }
        Value::GeometryCollection(gs) => {
            for sub in gs {
                let bb = geometry_bbox(sub)?;
                feed(bb[0], bb[1]);
                feed(bb[2], bb[3]);
            }
        }
    }
    if min[0].is_infinite() {
        return Err(anyhow!("no_geometry"));
    }
    Ok([min[0], min[1], max[0], max[1]])
}

fn parse_shapefile(path: &Path) -> Result<ParsedVector> {
    let mut reader = shapefile::Reader::from_path(path)?;
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut first_geom: Option<geojson::Geometry> = None;
    let mut count = 0u32;

    for res in reader.iter_shapes_and_records() {
        let (shape, _record) = res?;
        count += 1;
        match &shape {
            shapefile::Shape::Polygon(p) => {
                let bb = p.bbox();
                if bb.min.x < min[0] {
                    min[0] = bb.min.x
                }
                if bb.min.y < min[1] {
                    min[1] = bb.min.y
                }
                if bb.max.x > max[0] {
                    max[0] = bb.max.x
                }
                if bb.max.y > max[1] {
                    max[1] = bb.max.y
                }
                if first_geom.is_none() {
                    let rings: Vec<Vec<Vec<f64>>> = p
                        .rings()
                        .iter()
                        .map(|r| r.points().iter().map(|pt| vec![pt.x, pt.y]).collect())
                        .collect();
                    first_geom = Some(geojson::Geometry::new(geojson::Value::Polygon(rings)));
                }
            }
            shapefile::Shape::Point(p) => {
                if p.x < min[0] {
                    min[0] = p.x
                }
                if p.y < min[1] {
                    min[1] = p.y
                }
                if p.x > max[0] {
                    max[0] = p.x
                }
                if p.y > max[1] {
                    max[1] = p.y
                }
                if first_geom.is_none() {
                    first_geom = Some(geojson::Geometry::new(geojson::Value::Point(vec![
                        p.x, p.y,
                    ])));
                }
            }
            _ => {}
        }
    }
    if count == 0 || min[0].is_infinite() {
        return Err(anyhow!("no_geometry"));
    }
    Ok(ParsedVector {
        bbox: [min[0], min[1], max[0], max[1]],
        geometry: first_geom.ok_or_else(|| anyhow!("no_geometry"))?,
        layer_count: 1,
    })
}

fn parse_gpkg(path: &Path) -> Result<ParsedVector> {
    let conn = rusqlite::Connection::open(path)?;
    let row_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM gpkg_geometry_columns", [], |row| {
            row.get(0)
        })
        .map_err(|_| anyhow!("no_geometry"))?;
    let table: String = conn
        .query_row(
            "SELECT table_name FROM gpkg_geometry_columns LIMIT 1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| anyhow!("no_geometry"))?;
    let column: String = conn.query_row(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
        [&table],
        |row| row.get(0),
    )?;
    let blob: Vec<u8> = conn
        .query_row(
            &format!("SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1"),
            [],
            |row| row.get(0),
        )
        .map_err(|_| anyhow!("no_geometry"))?;
    if blob.len() < 8 || &blob[0..2] != b"GP" {
        return Err(anyhow!("not a GPKG geometry blob"));
    }
    let envelope_indicator = (blob[3] >> 1) & 0x07;
    let envelope_size = match envelope_indicator {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        _ => 0,
    };
    let header_len = 8 + envelope_size;
    let wkb = &blob[header_len..];
    let geom = parse_wkb(wkb)?;
    let bbox = geometry_bbox(&geom)?;
    Ok(ParsedVector {
        bbox,
        geometry: geom,
        layer_count: row_count as u32,
    })
}

fn parse_wkb(b: &[u8]) -> Result<geojson::Geometry> {
    if b.is_empty() {
        return Err(anyhow!("empty WKB"));
    }
    let little = b[0] == 1;
    let read_u32 = |i: usize| -> u32 {
        let arr: [u8; 4] = b[i..i + 4].try_into().unwrap();
        if little {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        }
    };
    let read_f64 = |i: usize| -> f64 {
        let arr: [u8; 8] = b[i..i + 8].try_into().unwrap();
        if little {
            f64::from_le_bytes(arr)
        } else {
            f64::from_be_bytes(arr)
        }
    };
    let typ = read_u32(1);
    if typ != 3 {
        return Err(anyhow!("only POLYGON supported in MVP, got type {}", typ));
    }
    let n_rings = read_u32(5) as usize;
    let mut p = 9usize;
    let mut rings: Vec<Vec<Vec<f64>>> = Vec::new();
    for _ in 0..n_rings {
        let n_pts = read_u32(p) as usize;
        p += 4;
        let mut ring = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            ring.push(vec![read_f64(p), read_f64(p + 8)]);
            p += 16;
        }
        rings.push(ring);
    }
    Ok(geojson::Geometry::new(geojson::Value::Polygon(rings)))
}

// ── GeoParquet support ───────────────────────────────────────────────────────
//
// GeoParquet stores geometries as WKB-encoded binary in a column whose name is
// declared in the file's `geo` key-value metadata (JSON). Spec ≥1.1 also
// stores per-column `bbox` metadata, which lets us short-circuit without
// decoding any geometry.
//
// We use Apache Arrow/Parquet to read columns; for the geometry payload we
// rely on the existing `parse_wkb()` only when the file is a single Polygon
// (legacy AOI use-case). For multi-feature files we just compute bbox by
// scanning WKB bytes for plausible lon/lat f64 pairs.

fn parse_geoparquet(path: &Path) -> Result<ParsedVector> {
    use arrow_array::Array;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| anyhow!("not a valid Parquet file: {e}"))?;

    // The Parquet KV metadata holds the GeoParquet `geo` JSON string.
    let geo_meta = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .as_ref()
        .and_then(|kvs| kvs.iter().find(|kv| kv.key == "geo"))
        .and_then(|kv| kv.value.as_ref())
        .ok_or_else(|| anyhow!("not a GeoParquet file (missing `geo` metadata)"))?;
    let geo: serde_json::Value = serde_json::from_str(geo_meta)
        .map_err(|e| anyhow!("invalid `geo` metadata JSON: {e}"))?;

    let primary_column = geo["primary_column"]
        .as_str()
        .ok_or_else(|| anyhow!("GeoParquet metadata has no primary_column"))?
        .to_string();

    // Reject non-WGS84 CRS — downstream tile math assumes EPSG:4326.
    // GeoParquet defaults to OGC:CRS84 (lon/lat WGS84) when crs is absent.
    if let Some(crs) = geo["columns"][&primary_column].get("crs") {
        if let Some(crs_obj) = crs.as_object() {
            // PROJJSON "id.code" or "id.authority"+"code" identifies the CRS.
            let code = crs_obj
                .get("id")
                .and_then(|i| i.get("code"))
                .and_then(|c| c.as_i64().or_else(|| c.as_str().and_then(|s| s.parse().ok())));
            if !matches!(code, None | Some(4326) | Some(4979)) {
                return Err(anyhow!(
                    "GeoParquet uses non-WGS84 CRS (code {:?}); reproject before importing",
                    code
                ));
            }
        }
    }

    // Fast path: column-level bbox is in metadata (GeoParquet 1.1+).
    let total_rows = builder.metadata().file_metadata().num_rows() as u32;
    if let Some(bbox_arr) = geo["columns"][&primary_column].get("bbox").and_then(|b| b.as_array()) {
        if bbox_arr.len() == 4 {
            let xs: Vec<f64> = bbox_arr.iter().filter_map(|v| v.as_f64()).collect();
            if xs.len() == 4 {
                let bbox = [xs[0], xs[1], xs[2], xs[3]];
                return Ok(ParsedVector {
                    bbox,
                    geometry: bbox_polygon(bbox),
                    layer_count: total_rows,
                });
            }
        }
    }

    // Slow path: stream the geometry column and scan WKB bytes for lon/lat pairs.
    let reader = builder.build().map_err(|e| anyhow!("parquet read: {e}"))?;
    let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let mut count: u32 = 0;

    for batch in reader {
        let batch = batch.map_err(|e| anyhow!("parquet batch: {e}"))?;
        let col = batch
            .column_by_name(&primary_column)
            .ok_or_else(|| anyhow!("column `{}` missing from batch", primary_column))?;
        let bin = col
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .ok_or_else(|| anyhow!("geometry column is not Binary; large-binary not yet handled"))?;
        for i in 0..bin.len() {
            if bin.is_null(i) {
                continue;
            }
            wkb_extents_into(bin.value(i), &mut bbox);
            count += 1;
        }
    }

    if !bbox[0].is_finite() {
        return Err(anyhow!("GeoParquet contains no readable geometries"));
    }
    Ok(ParsedVector {
        bbox,
        geometry: bbox_polygon(bbox),
        layer_count: count,
    })
}

/// Build a closed-ring polygon for a bbox `[minx, miny, maxx, maxy]`.
fn bbox_polygon(b: [f64; 4]) -> geojson::Geometry {
    let ring = vec![
        vec![b[0], b[1]],
        vec![b[2], b[1]],
        vec![b[2], b[3]],
        vec![b[0], b[3]],
        vec![b[0], b[1]],
    ];
    geojson::Geometry::new(geojson::Value::Polygon(vec![ring]))
}

/// Walk a WKB byte stream and update `bbox` with every lon/lat point we find.
/// Supports the common ISO WKB types (Point/LineString/Polygon and the Multi
/// variants, plus GeometryCollection); ignores unknown types rather than
/// failing — the goal is bbox coverage, not perfect geometry decoding.
fn wkb_extents_into(b: &[u8], bbox: &mut [f64; 4]) {
    let mut p = 0usize;
    walk_wkb(b, &mut p, bbox);
}

fn walk_wkb(b: &[u8], p: &mut usize, bbox: &mut [f64; 4]) -> Option<()> {
    if *p + 5 > b.len() {
        return None;
    }
    let little = b[*p] == 1;
    let read_u32 = |b: &[u8], i: usize| -> u32 {
        let arr: [u8; 4] = b[i..i + 4].try_into().ok().unwrap_or([0; 4]);
        if little {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        }
    };
    let read_f64 = |b: &[u8], i: usize| -> f64 {
        let arr: [u8; 8] = b[i..i + 8].try_into().ok().unwrap_or([0; 8]);
        if little {
            f64::from_le_bytes(arr)
        } else {
            f64::from_be_bytes(arr)
        }
    };
    // Strip ISO WKB Z/M flags (0x80000000, 0x40000000) and EWKB SRID flag (0x20000000).
    let raw_type = read_u32(b, *p + 1);
    let typ = raw_type & 0xFF;
    let has_srid = raw_type & 0x20000000 != 0;
    *p += 5;
    if has_srid {
        *p += 4;
    }

    let feed = |x: f64, y: f64, bbox: &mut [f64; 4]| {
        if x < bbox[0] {
            bbox[0] = x;
        }
        if y < bbox[1] {
            bbox[1] = y;
        }
        if x > bbox[2] {
            bbox[2] = x;
        }
        if y > bbox[3] {
            bbox[3] = y;
        }
    };

    match typ {
        1 => {
            // POINT
            if *p + 16 <= b.len() {
                let (x, y) = (read_f64(b, *p), read_f64(b, *p + 8));
                feed(x, y, bbox);
                *p += 16;
            }
        }
        2 => {
            // LINESTRING
            let n = read_u32(b, *p) as usize;
            *p += 4;
            for _ in 0..n {
                if *p + 16 > b.len() {
                    return None;
                }
                feed(read_f64(b, *p), read_f64(b, *p + 8), bbox);
                *p += 16;
            }
        }
        3 => {
            // POLYGON
            let n_rings = read_u32(b, *p) as usize;
            *p += 4;
            for _ in 0..n_rings {
                if *p + 4 > b.len() {
                    return None;
                }
                let n_pts = read_u32(b, *p) as usize;
                *p += 4;
                for _ in 0..n_pts {
                    if *p + 16 > b.len() {
                        return None;
                    }
                    feed(read_f64(b, *p), read_f64(b, *p + 8), bbox);
                    *p += 16;
                }
            }
        }
        4..=7 => {
            // MULTIPOINT (4) | MULTILINESTRING (5) | MULTIPOLYGON (6) | GEOMETRYCOLLECTION (7)
            let n = read_u32(b, *p) as usize;
            *p += 4;
            for _ in 0..n {
                walk_wkb(b, p, bbox)?;
            }
        }
        _ => return None, // unknown type
    }
    Some(())
}

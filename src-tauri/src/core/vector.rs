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

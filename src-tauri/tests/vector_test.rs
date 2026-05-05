use imagery_downloader_lib::core::vector::{parse_vector, ParsedVector};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn parse_geojson_polygon() {
    let p = fixture("triangle.geojson");
    let r: ParsedVector = parse_vector(&p).unwrap();
    assert_eq!(r.layer_count, 1);
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
}

#[test]
fn geojson_feature_collection_layer_count() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("multi.geojson");
    std::fs::write(
        &p,
        r#"{
        "type":"FeatureCollection",
        "features":[
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,0]}},
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,1]}}
        ]
    }"#,
    )
    .unwrap();
    let r = parse_vector(&p).unwrap();
    assert_eq!(r.layer_count, 2);
}

fn build_wkb_polygon(pts: &[[f64; 2]]) -> Vec<u8> {
    let mut w = Vec::new();
    w.push(1u8); // little-endian
    w.extend_from_slice(&3u32.to_le_bytes()); // type = Polygon
    w.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
    w.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for p in pts {
        w.extend_from_slice(&p[0].to_le_bytes());
        w.extend_from_slice(&p[1].to_le_bytes());
    }
    w
}

fn make_gpkg_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    use rusqlite::Connection;
    let path = dir.join("triangle.gpkg");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        r#"
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL, m TINYINT NOT NULL
        );
        CREATE TABLE features (id INTEGER PRIMARY KEY, geom BLOB);
        INSERT INTO gpkg_geometry_columns VALUES ('features', 'geom', 'POLYGON', 4326, 0, 0);
        "#,
    )
    .unwrap();

    let wkb = build_wkb_polygon(&[[100.0, 30.0], [102.0, 30.0], [101.0, 32.0], [100.0, 30.0]]);
    let mut blob = vec![0x47, 0x50, 0x00, 0x00]; // 'G','P', version=0, flags=0 (no envelope)
    blob.extend_from_slice(&4326i32.to_le_bytes());
    blob.extend_from_slice(&wkb);
    conn.execute("INSERT INTO features (geom) VALUES (?)", [&blob])
        .unwrap();
    path
}

#[test]
fn parse_gpkg_polygon() {
    let dir = tempfile::tempdir().unwrap();
    let p = make_gpkg_fixture(dir.path());
    let r = parse_vector(&p).unwrap();
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
    assert_eq!(r.layer_count, 1);
}

/// Build a minimum-viable GeoParquet file in a temp dir.
/// Writes a single Binary column "geometry" with one WKB polygon, and stamps
/// the `geo` key in the file's KV metadata so parse_geoparquet recognises it.
fn make_geoparquet_fixture(
    dir: &std::path::Path,
    geo_meta: &str,
    pts: &[[f64; 2]],
) -> std::path::PathBuf {
    use arrow_array::{ArrayRef, BinaryArray, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use parquet::format::KeyValue;
    use std::fs::File;
    use std::sync::Arc;

    let path = dir.join("aoi.parquet");
    let wkb = build_wkb_polygon(pts);
    let geom_array: ArrayRef = Arc::new(BinaryArray::from_vec(vec![&wkb]));
    let batch = RecordBatch::try_from_iter(vec![("geometry", geom_array)]).unwrap();

    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue {
            key: "geo".to_string(),
            value: Some(geo_meta.to_string()),
        }]))
        .build();

    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

#[test]
fn parse_geoparquet_with_metadata_bbox() {
    let dir = tempfile::tempdir().unwrap();
    // GeoParquet 1.1 column-level bbox in metadata → fast path, no WKB scan.
    let geo = r#"{
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "bbox": [100.0, 30.0, 102.0, 32.0]
            }
        }
    }"#;
    let p = make_geoparquet_fixture(
        dir.path(),
        geo,
        &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], // intentionally different from bbox
    );
    let r = parse_vector(&p).unwrap();
    // Reads metadata bbox, NOT the WKB — proves fast path is taken.
    assert!((r.bbox[0] - 100.0).abs() < 1e-9, "bbox.minx={}", r.bbox[0]);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
    assert_eq!(r.layer_count, 1);
}

#[test]
fn parse_geoparquet_falls_back_to_wkb_scan() {
    let dir = tempfile::tempdir().unwrap();
    // No bbox in metadata → must scan the WKB column.
    let geo = r#"{
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": { "geometry": { "encoding": "WKB", "geometry_types": ["Polygon"] } }
    }"#;
    let p = make_geoparquet_fixture(
        dir.path(),
        geo,
        &[[100.0, 30.0], [102.0, 30.0], [101.0, 32.0], [100.0, 30.0]],
    );
    let r = parse_vector(&p).unwrap();
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert!((r.bbox[1] - 30.0).abs() < 1e-9);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
    assert!((r.bbox[3] - 32.0).abs() < 1e-9);
    assert_eq!(r.layer_count, 1);
}

#[test]
fn parse_geoparquet_rejects_non_wgs84() {
    let dir = tempfile::tempdir().unwrap();
    // CRS is EPSG:3857 (web mercator) — must error out.
    let geo = r#"{
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": { "id": { "authority": "EPSG", "code": 3857 } }
            }
        }
    }"#;
    let p = make_geoparquet_fixture(
        dir.path(),
        geo,
        &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
    );
    let err = parse_vector(&p).unwrap_err().to_string();
    assert!(err.contains("non-WGS84"), "got: {err}");
}

#[test]
fn parse_geoparquet_rejects_plain_parquet() {
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.parquet");

    // Parquet without GeoParquet metadata → friendly error, not a panic.
    let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    let batch = RecordBatch::try_from_iter(vec![("x", arr)]).unwrap();
    let file = File::create(&path).unwrap();
    let mut w = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let err = parse_vector(&path).unwrap_err().to_string();
    assert!(err.contains("not a GeoParquet"), "got: {err}");
}

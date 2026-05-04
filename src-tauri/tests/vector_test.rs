use imagery_downloader_lib::core::vector::{parse_vector, ParsedVector};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
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

    let wkb = build_wkb_polygon(&[
        [100.0, 30.0],
        [102.0, 30.0],
        [101.0, 32.0],
        [100.0, 30.0],
    ]);
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

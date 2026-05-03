use imagery_downloader_lib::history::{Store, HistoryEntry};
use std::path::PathBuf;

fn tmp_path() -> PathBuf {
    let p = std::env::temp_dir().join(format!("history-{}.json", uuid::Uuid::new_v4()));
    p
}

fn entry(zoom: u32) -> HistoryEntry {
    HistoryEntry {
        bbox: [100.0, 30.0, 110.0, 40.0],
        zoom,
        source: "esri".into(),
        output_path: "/tmp/x.tif".into(),
        ok: true,
        duration_sec: 5.0,
        total_tiles: 100,
        failed_tiles: 0,
        output_size_mb: 5.0,
        finished_at: "2026-05-03T10:00:00Z".into(),
    }
}

#[test]
fn roundtrip_empty() {
    let s = Store::open(tmp_path()).unwrap();
    assert!(s.list().is_empty());
}

#[test]
fn add_and_list() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    s.add(entry(18)).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 2);
    // newest first
    assert_eq!(l[0].zoom, 18);
}

#[test]
fn dedupe_by_bbox_zoom_source() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    let mut e = entry(17);
    e.finished_at = "2026-05-03T11:00:00Z".into();
    s.add(e).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].finished_at, "2026-05-03T11:00:00Z");
}

#[test]
fn caps_at_10() {
    let s = Store::open(tmp_path()).unwrap();
    for z in 8..=22 { s.add(entry(z)).unwrap(); }
    let l = s.list();
    assert_eq!(l.len(), 10);
    // newest still on top
    assert_eq!(l[0].zoom, 22);
}

#[test]
fn clear() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    s.clear().unwrap();
    assert!(s.list().is_empty());
}

#[test]
fn persists_across_open() {
    let p = tmp_path();
    {
        let s = Store::open(&p).unwrap();
        s.add(entry(17)).unwrap();
    }
    let s2 = Store::open(&p).unwrap();
    assert_eq!(s2.list().len(), 1);
}

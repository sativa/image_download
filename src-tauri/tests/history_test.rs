use imagery_downloader_lib::core::history::{HistoryEntry, Store};
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
        output_path: "/tmp/yuzhong".into(), // user-chosen folder, not a .tif file
        ok: true,
        duration_sec: 5.0,
        total_tiles: 100,
        failed_tiles: 0,
        output_size_mb: 5.0,
        finished_at: "2026-05-03T10:00:00Z".into(),
        completed_tiles: 100,
        status: HistoryEntry::STATUS_COMPLETED.into(),
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
    for z in 8..=22 {
        s.add(entry(z)).unwrap();
    }
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

#[test]
fn legacy_history_without_status_loads_as_completed() {
    // Pre-existing history.json (e.g. from an older app build) doesn't have
    // `status` or `completed_tiles` fields. serde defaults must rehydrate it
    // without panicking.
    let p = tmp_path();
    let legacy = r#"[{
        "bbox":[100.0,30.0,110.0,40.0],"zoom":17,"source":"esri",
        "output_path":"/tmp/x.tif","ok":true,"duration_sec":5.0,
        "total_tiles":100,"failed_tiles":0,"output_size_mb":5.0,
        "finished_at":"2026-05-03T10:00:00Z"
    }]"#;
    std::fs::write(&p, legacy).unwrap();
    let s = Store::open(&p).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].status, HistoryEntry::STATUS_COMPLETED);
    assert_eq!(l[0].completed_tiles, 0); // unknown for old entries
}

#[test]
fn in_progress_entry_is_replaced_by_completed_entry() {
    // The Store dedupes by (bbox, zoom, source). When a task progresses
    // from in_progress → completed, the new row replaces the old one.
    let s = Store::open(tmp_path()).unwrap();
    let mut e = entry(17);
    e.completed_tiles = 30;
    e.status = HistoryEntry::STATUS_IN_PROGRESS.into();
    s.add(e).unwrap();
    assert_eq!(s.list()[0].status, HistoryEntry::STATUS_IN_PROGRESS);

    let mut done = entry(17);
    done.completed_tiles = 100;
    done.status = HistoryEntry::STATUS_COMPLETED.into();
    s.add(done).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 1, "old in_progress row must be replaced, not duplicated");
    assert_eq!(l[0].status, HistoryEntry::STATUS_COMPLETED);
    assert_eq!(l[0].completed_tiles, 100);
}

#[test]
fn legacy_output_path_with_tif_suffix_is_migrated_to_folder() {
    // Reproduces the yuzhong nested-folder bug: history.json from an older
    // build wrote `output_path` as the final .tif file. On Store::open we
    // detect those entries and rewrite them to the parent folder so a
    // future restore feeds the directory back into input.outputPath.
    let p = tmp_path();
    let polluted = r#"[{
        "bbox":[100.0,30.0,110.0,40.0],"zoom":17,"source":"google",
        "output_path":"/Users/me/Downloads/yuzhong/imagery_z17_google_9c402df0.tif",
        "ok":false,"duration_sec":0.0,"total_tiles":89760,"failed_tiles":0,
        "output_size_mb":0.0,"finished_at":"epoch:1714809600",
        "completed_tiles":1342,"status":"in_progress"
    }, {
        "bbox":[0.0,0.0,1.0,1.0],"zoom":10,"source":"esri",
        "output_path":"/Users/me/Downloads/test.TIFF",
        "ok":true,"duration_sec":5.0,"total_tiles":4,"failed_tiles":0,
        "output_size_mb":1.0,"finished_at":"epoch:1714809600",
        "completed_tiles":4,"status":"completed"
    }]"#;
    std::fs::write(&p, polluted).unwrap();

    let s = Store::open(&p).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 2);
    // Both .tif and .TIFF (case-insensitive) get rewritten to their parent.
    assert_eq!(l[0].output_path, "/Users/me/Downloads/yuzhong");
    assert_eq!(l[1].output_path, "/Users/me/Downloads");

    // Migration should have been persisted, so re-opening reads back the
    // already-rewritten entries (no second migration needed).
    let s2 = Store::open(&p).unwrap();
    let l2 = s2.list();
    assert_eq!(l2[0].output_path, "/Users/me/Downloads/yuzhong");
    assert_eq!(l2[1].output_path, "/Users/me/Downloads");
}

#[test]
fn legacy_double_nested_tif_path_is_fully_stripped() {
    // Reproduces the exact yuzhong shape: the nested-folder bug produced
    // paths like `…/yuzhong/imagery.tif/imagery.tif` (or even three
    // layers). Migration must keep stripping until what remains is a
    // real directory, not a half-fixed `…/imagery.tif/`.
    let p = tmp_path();
    let polluted = r#"[{
        "bbox":[103.66,35.39,104.57,36.0],"zoom":17,"source":"google",
        "output_path":"/Users/me/Downloads/yuzhong/imagery_z17_google_9c402df0.tif/imagery_z17_google_9c402df0.tif",
        "ok":false,"duration_sec":0.0,"total_tiles":89760,"failed_tiles":0,
        "output_size_mb":0.0,"finished_at":"epoch:1714809600",
        "completed_tiles":40744,"status":"in_progress"
    }]"#;
    std::fs::write(&p, polluted).unwrap();
    let s = Store::open(&p).unwrap();
    assert_eq!(
        s.list()[0].output_path,
        "/Users/me/Downloads/yuzhong",
        "two layers of `.tif` should be stripped down to the actual folder"
    );
}

#[test]
fn folder_paths_are_left_alone_by_migration() {
    // A correctly-formed entry (output_path is a directory) must not be
    // rewritten to its parent.
    let p = tmp_path();
    let s = Store::open(&p).unwrap();
    let mut e = entry(17);
    e.output_path = "/Users/me/Downloads/yuzhong".into();
    s.add(e).unwrap();
    drop(s);

    let s2 = Store::open(&p).unwrap();
    assert_eq!(s2.list()[0].output_path, "/Users/me/Downloads/yuzhong");
}

from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ValidationError

from .data.vhd11k import Sample, concept_taxonomy, load_vhd11k
from .metrics.scoring import read_predictions
from .risk_coverage import logit
from .runner import load_clean
from .val_gate import extract_vlm_logits, load_split_aligned


class LabelRecord(BaseModel):
    slot: int
    sample_id: str
    is_sentinel: bool = False
    harmful: bool | None = None
    borderline: bool = False
    skipped: bool = False
    rater: str = "anon"
    elapsed_ms: float = 0.0
    timestamp_utc: str = ""


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Compute binary Cohen's kappa between two aligned boolean label sequences."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa_yes = sum(a) / n
    pb_yes = sum(b) / n
    pe = pa_yes * pb_yes + (1.0 - pa_yes) * (1.0 - pb_yes)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(1.0 - po) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def read_labels(path: Path) -> dict[int, LabelRecord]:
    """Read crash-resumable JSONL label log; later entries for the same slot win (supports undo)."""
    records: dict[int, LabelRecord] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = LabelRecord.model_validate_json(line)
        except ValidationError:
            continue
        records[rec.slot] = rec
    return records


def build_val_audit_manifest(
    dataset: Path,
    splits_path: Path,
    gate_path: Path,
    probe_path: Path,
    vlm_path: Path,
    n_in_band_control: int = 30,
    n_outer_control: int = 30,
    n_sentinels: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a blind, shuffled val audit manifest combining all both-wrong in-band cases with controls."""
    gate = json.loads(gate_path.expanduser().read_text(encoding="utf-8"))
    op = gate["locked_operating_point"]
    c_star = float(op["c_star"])
    z_cut = float(op["z_vlm_cut_star"])

    aligned = load_split_aligned(
        dataset=dataset,
        splits_path=splits_path,
        split_name="val",
        probe_path=probe_path,
        vlm_path=vlm_path,
        components_path=None,
    )
    if aligned["splits_sha256"] != gate["splits_sha256"]:
        raise SystemExit("ERROR: splits_sha256 mismatch between --splits and --gate.")

    split_ids: list[str] = aligned["split_ids"]
    y_true: np.ndarray = aligned["y_true"]
    p_probe: np.ndarray = aligned["p_probe"]
    z_vlm: np.ndarray = aligned["z_vlm"]

    band = np.abs(logit(p_probe)) < c_star
    pred_probe = p_probe >= 0.50
    pred_vlm = z_vlm >= z_cut

    prb_ok = pred_probe == y_true
    vlm_ok = pred_vlm == y_true

    both_wrong_idx = [int(i) for i in np.where(band & (~prb_ok) & (~vlm_ok))[0]]
    in_band_ctrl_pool = [int(i) for i in np.where(band & (prb_ok | vlm_ok))[0]]
    outer_ctrl_pool = [int(i) for i in np.where(~band)[0]]

    rng = np.random.default_rng(seed)
    chosen_in_ctrl = sorted(
        int(i)
        for i in rng.choice(in_band_ctrl_pool, size=min(n_in_band_control, len(in_band_ctrl_pool)), replace=False)
    )
    chosen_outer_ctrl = sorted(
        int(i) for i in rng.choice(outer_ctrl_pool, size=min(n_outer_control, len(outer_ctrl_pool)), replace=False)
    )

    items: list[dict[str, Any]] = []
    for idx in both_wrong_idx:
        items.append({"sample_id": split_ids[idx], "stratum": "in_band_both_wrong", "is_sentinel": False})
    for idx in chosen_in_ctrl:
        items.append({"sample_id": split_ids[idx], "stratum": "in_band_control", "is_sentinel": False})
    for idx in chosen_outer_ctrl:
        items.append({"sample_id": split_ids[idx], "stratum": "outer_resolved_control", "is_sentinel": False})

    # Shuffle primary items first, then insert hidden sentinel repeats in the second half for drift check.
    order = rng.permutation(len(items)).tolist()
    shuffled = [items[i] for i in order]

    if n_sentinels > 0 and len(shuffled) >= 4:
        first_half = max(1, len(shuffled) // 2)
        k_sent = min(n_sentinels, first_half)
        sent_sources = [shuffled[int(i)] for i in rng.choice(first_half, size=k_sent, replace=False)]
        sent_items = [
            {"sample_id": it["sample_id"], "stratum": it["stratum"], "is_sentinel": True} for it in sent_sources
        ]
        # Interleave sentinels randomly throughout the second half so repeats are well separated.
        second_half = shuffled[first_half:] + sent_items
        sec_order = rng.permutation(len(second_half)).tolist()
        shuffled = shuffled[:first_half] + [second_half[i] for i in sec_order]

    for slot, item in enumerate(shuffled):
        item["slot"] = slot

    return {
        "mode": "val-audit",
        "split": "val",
        "splits_sha256": aligned["splits_sha256"],
        "gate_path": str(gate_path),
        "c_star": c_star,
        "z_vlm_cut_star": z_cut,
        "seed": seed,
        "counts": {
            "in_band_both_wrong": len(both_wrong_idx),
            "in_band_control": len(chosen_in_ctrl),
            "outer_resolved_control": len(chosen_outer_ctrl),
            "sentinels": sum(1 for it in shuffled if it["is_sentinel"]),
            "total_slots": len(shuffled),
            "unique_images": len({it["sample_id"] for it in shuffled}),
        },
        "items": shuffled,
    }


def build_split_manifest(
    splits_path: Path,
    split_name: Literal["train", "val", "test"],
    n_sentinels: int = 30,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a blind shuffled labeling manifest over an entire split (for multi-week labeling)."""
    splits_raw = splits_path.expanduser().read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    split_ids: list[str] = list(json.loads(splits_raw)[split_name])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(split_ids)).tolist()
    chosen_ids = [split_ids[i] for i in perm[:limit]]

    shuffled = [{"sample_id": sid, "stratum": split_name, "is_sentinel": False} for sid in chosen_ids]
    if n_sentinels > 0 and len(shuffled) >= 4:
        first_half = max(1, len(shuffled) // 2)
        k_sent = min(n_sentinels, first_half)
        sent_sources = [shuffled[int(i)] for i in rng.choice(first_half, size=k_sent, replace=False)]
        sent_items = [{"sample_id": it["sample_id"], "stratum": split_name, "is_sentinel": True} for it in sent_sources]
        second_half = shuffled[first_half:] + sent_items
        sec_order = rng.permutation(len(second_half)).tolist()
        shuffled = shuffled[:first_half] + [second_half[i] for i in sec_order]

    for slot, item in enumerate(shuffled):
        item["slot"] = slot

    return {
        "mode": "split",
        "split": split_name,
        "splits_sha256": splits_sha256,
        "seed": seed,
        "counts": {
            "unique_images": len(chosen_ids),
            "sentinels": sum(1 for it in shuffled if it["is_sentinel"]),
            "total_slots": len(shuffled),
        },
        "items": shuffled,
    }


_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>disRated Blind Audit</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #eceff4; display: flex; flex-direction: column; height: 100vh; }
  header { display: flex; justify-content: space-between; align-items: center; padding: 10px 18px; background: #1a1e26; border-bottom: 1px solid #2b3240; font-size: 14px; }
  .bar-wrap { flex: 1; max-width: 360px; height: 8px; background: #2b3240; border-radius: 99px; margin: 0 16px; overflow: hidden; }
  .bar { height: 100%; background: #4c9aff; width: 0%; transition: width 0.15s ease; }
  main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 16px; position: relative; }
  main img { max-width: 96vw; max-height: calc(100vh - 175px); object-fit: contain; border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,0.45); background: #0a0c10; }
  .done-msg { text-align: center; font-size: 22px; color: #8be9fd; }
  footer { padding: 12px 18px 16px; background: #1a1e26; border-top: 1px solid #2b3240; display: flex; flex-direction: column; gap: 10px; align-items: center; }
  .controls { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  button { border: 1px solid #3b4456; border-radius: 8px; padding: 10px 18px; font-size: 15px; font-weight: 600; cursor: pointer; background: #242b38; color: #eceff4; }
  button:hover { filter: brightness(1.12); }
  .safe { background: #1c5c3a; border-color: #2c8a57; }
  .harm { background: #7f232b; border-color: #b8333f; }
  .borderline.active { background: #9c6b15; border-color: #e2a42c; color: #fff; }
  .hint { font-size: 12px; color: #9aa5b8; }
  kbd { background: #2b3240; padding: 2px 6px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 12px; color: #e2e8f0; }
</style>
</head>
<body>
<header>
  <div><strong>disRated Blind Labeling</strong> &middot; <span id="rater"></span></div>
  <div class="bar-wrap"><div class="bar" id="bar"></div></div>
  <div id="status">Loading...</div>
</header>
<main id="stage">
  <img id="img" alt="blind audit image" />
</main>
<footer>
  <div class="controls">
    <button class="safe" onclick="submitLabel(false, false)">Safe (<kbd>A</kbd> / <kbd>1</kbd>)</button>
    <button class="harm" onclick="submitLabel(true, false)">Harmful (<kbd>L</kbd> / <kbd>2</kbd>)</button>
    <button class="borderline" id="btnBorderline" onclick="toggleBorderline()">Borderline: OFF (<kbd>Space</kbd>)</button>
    <button onclick="submitLabel(null, true)">Skip (<kbd>S</kbd>)</button>
    <button onclick="undoLast()">Undo (<kbd>Z</kbd>)</button>
  </div>
  <div class="hint">Blind mode: filenames, VHD11K judge labels, and model scores are hidden. Progress auto-saves after every image.</div>
</footer>
<script>
let current = null;
let borderline = false;
let shownAt = performance.now();

function updateBorderlineBtn() {
  const b = document.getElementById("btnBorderline");
  b.classList.toggle("active", borderline);
  b.innerHTML = `Borderline: ${borderline ? "ON" : "OFF"} (<kbd>Space</kbd>)`;
}

function toggleBorderline() {
  borderline = !borderline;
  updateBorderlineBtn();
}

async function fetchState() {
  const res = await fetch("/api/state");
  const st = await res.json();
  document.getElementById("rater").textContent = `rater=${st.rater}`;
  const pct = st.total > 0 ? Math.round((st.done / st.total) * 100) : 100;
  document.getElementById("bar").style.width = `${pct}%`;
  document.getElementById("status").textContent = `${st.done} / ${st.total} (${pct}%)`;
  const stage = document.getElementById("stage");
  if (st.next_slot === null) {
    current = null;
    stage.innerHTML = `<div class="done-msg">All ${st.total} items completed!<br/><small>You can close this tab and run <code>disrated label score</code>.</small></div>`;
    return;
  }
  current = st.next_slot;
  borderline = false;
  updateBorderlineBtn();
  shownAt = performance.now();
  let img = document.getElementById("img");
  if (!img) {
    stage.innerHTML = '<img id="img" alt="blind audit image" />';
    img = document.getElementById("img");
  }
  img.src = `/img/${current}?t=${Date.now()}`;
}

async function submitLabel(harmful, skipped) {
  if (current === null) return;
  const elapsed_ms = Math.round(performance.now() - shownAt);
  await fetch("/api/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slot: current, harmful, borderline, skipped, elapsed_ms }),
  });
  await fetchState();
}

async function undoLast() {
  await fetch("/api/undo", { method: "POST" });
  await fetchState();
}

window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === "a" || k === "1") { e.preventDefault(); submitLabel(false, false); }
  else if (k === "l" || k === "2") { e.preventDefault(); submitLabel(true, false); }
  else if (k === " " || k === "b") { e.preventDefault(); toggleBorderline(); }
  else if (k === "s") { e.preventDefault(); submitLabel(null, true); }
  else if (k === "z" || k === "backspace") { e.preventDefault(); undoLast(); }
});

fetchState();
</script>
</body>
</html>
"""


def _render_clean_jpeg_bytes(path: Path) -> bytes:
    """Decode through load_clean() (pixels only, zero metadata) and encode to JPEG bytes in memory."""
    img = load_clean(path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def serve_blind_ui(
    dataset: Path,
    manifest_path: Path,
    out_path: Path,
    rater: str = "vor",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve a local zero-dependency blind labeling UI reading stripped pixels via load_clean()."""
    manifest = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = manifest["items"]
    samples, _ = load_vhd11k(dataset.expanduser())
    by_id: dict[str, Sample] = {s.sample_id: s for s in samples}

    missing = [it["sample_id"] for it in items if it["sample_id"] not in by_id]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} manifest image(s) missing in {dataset} (first: {missing[0]}).")

    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    done_map: dict[int, LabelRecord] = read_labels(out_path)
    history: list[int] = sorted(done_map.keys())

    def _next_slot() -> int | None:
        for it in items:
            slot = int(it["slot"])
            if slot not in done_map:
                return slot
        return None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], code: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path_only = self.path.split("?", 1)[0]
            if path_only == "/":
                raw = _HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            if path_only == "/api/state":
                with lock:
                    st = {
                        "rater": rater,
                        "total": len(items),
                        "done": len(done_map),
                        "next_slot": _next_slot(),
                    }
                self._send_json(st)
                return

            if path_only.startswith("/img/"):
                try:
                    slot = int(path_only[len("/img/") :])
                    item = items[slot]
                except (ValueError, IndexError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                raw = _render_clean_jpeg_bytes(by_id[item["sample_id"]].path)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path_only = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            if path_only == "/api/label":
                slot = int(body["slot"])
                item = items[slot]
                rec = LabelRecord(
                    slot=slot,
                    sample_id=item["sample_id"],
                    is_sentinel=bool(item.get("is_sentinel", False)),
                    harmful=body.get("harmful"),
                    borderline=bool(body.get("borderline", False)),
                    skipped=bool(body.get("skipped", False)),
                    rater=rater,
                    elapsed_ms=float(body.get("elapsed_ms", 0.0)),
                    timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                )
                with lock:
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(rec.model_dump_json() + "\n")
                        f.flush()
                    done_map[slot] = rec
                    if slot in history:
                        history.remove(slot)
                    history.append(slot)
                self._send_json({"ok": True, "done": len(done_map), "total": len(items)})
                return

            if path_only == "/api/undo":
                with lock:
                    if history:
                        last_slot = history.pop()
                        done_map.pop(last_slot, None)
                        # Rewrite clean snapshot without the undone slot.
                        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                        tmp.write_text(
                            "".join(done_map[s].model_dump_json() + "\n" for s in sorted(done_map)),
                            encoding="utf-8",
                        )
                        tmp.replace(out_path)
                self._send_json({"ok": True, "done": len(done_map), "total": len(items)})
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(
        f"Blind labeling UI running at {url}  (manifest={manifest_path}, out={out_path}, done={len(done_map)}/{len(items)})"
    )
    print("Press Ctrl+C to stop. Progress is saved after every image.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. Saved {len(done_map)}/{len(items)} labels to {out_path}.")
    finally:
        server.server_close()


def score_human_audit(
    dataset: Path,
    manifest_path: Path,
    rater_paths: list[Path],
    probe_path: Path | None = None,
    vlm_specs: list[tuple[str, Path, float]] | None = None,
) -> dict[str, Any]:
    """Score one or more blind human label files against VHD11K judge and optional models."""
    manifest = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = manifest["items"]
    samples, _ = load_vhd11k(dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}

    # Primary (non-sentinel) items and sentinel repeat pairs
    primary_items = [it for it in items if not it.get("is_sentinel", False)]
    sentinel_items = [it for it in items if it.get("is_sentinel", False)]
    first_slot_by_sid = {it["sample_id"]: int(it["slot"]) for it in primary_items}

    rater_maps: dict[str, dict[int, LabelRecord]] = {}
    for rp in rater_paths:
        recs = read_labels(rp.expanduser())
        rater_name = next((r.rater for r in recs.values() if r.rater), rp.stem)
        if rater_name in rater_maps:
            rater_name = rp.stem
        rater_maps[rater_name] = recs

    # 1. Per-rater completion, borderline rate, intra-rater drift (sentinels), and judge agreement
    per_rater: dict[str, Any] = {}
    for rname, rmap in rater_maps.items():
        valid_prim = [
            (it, rmap[int(it["slot"])])
            for it in primary_items
            if int(it["slot"]) in rmap
            and not rmap[int(it["slot"])].skipped
            and rmap[int(it["slot"])].harmful is not None
        ]
        n_skipped = sum(1 for it in primary_items if int(it["slot"]) in rmap and rmap[int(it["slot"])].skipped)
        by_stratum: dict[str, dict[str, Any]] = {}
        strata = sorted({it["stratum"] for it in primary_items})
        for st in strata:
            sub = [(it, rec) for it, rec in valid_prim if it["stratum"] == st]
            if not sub:
                continue
            agree = [bool(rec.harmful) == by_id[it["sample_id"]].harmful for it, rec in sub]
            bdr = [bool(rec.borderline) for _, rec in sub]
            by_stratum[st] = {
                "n": len(sub),
                "judge_agreement": float(np.mean(agree)),
                "judge_disagreement": float(1.0 - np.mean(agree)),
                "borderline_rate": float(np.mean(bdr)),
                "human_harm_rate": float(np.mean([bool(rec.harmful) for _, rec in sub])),
            }

        # Taxonomy breakdown (concept cohort only)
        by_tax: dict[str, dict[str, Any]] = {}
        for tax in ("concrete_harm", "contextual_normative"):
            sub = [(it, rec) for it, rec in valid_prim if concept_taxonomy(by_id[it["sample_id"]].concept) == tax]
            if not sub:
                continue
            agree = [bool(rec.harmful) == by_id[it["sample_id"]].harmful for it, rec in sub]
            by_tax[tax] = {
                "n": len(sub),
                "judge_agreement": float(np.mean(agree)),
                "borderline_rate": float(np.mean([bool(rec.borderline) for _, rec in sub])),
            }

        # Intra-rater consistency on hidden sentinel repeats
        sent_a, sent_b = [], []
        for sit in sentinel_items:
            s_slot = int(sit["slot"])
            p_slot = first_slot_by_sid.get(sit["sample_id"])
            if p_slot is None or s_slot not in rmap or p_slot not in rmap:
                continue
            r1, r2 = rmap[p_slot], rmap[s_slot]
            if not r1.skipped and not r2.skipped and r1.harmful is not None and r2.harmful is not None:
                sent_a.append(bool(r1.harmful))
                sent_b.append(bool(r2.harmful))

        overall_agree = [bool(rec.harmful) == by_id[it["sample_id"]].harmful for it, rec in valid_prim]
        per_rater[rname] = {
            "n_primary_labeled": len(valid_prim),
            "n_primary_total": len(primary_items),
            "n_skipped": n_skipped,
            "overall_judge_agreement": float(np.mean(overall_agree)) if overall_agree else None,
            "overall_borderline_rate": float(np.mean([bool(rec.borderline) for _, rec in valid_prim]))
            if valid_prim
            else None,
            "intra_rater_sentinels": {
                "n": len(sent_a),
                "agreement": float(np.mean([x == y for x, y in zip(sent_a, sent_b, strict=True)])) if sent_a else None,
                "kappa": cohens_kappa(sent_a, sent_b) if sent_a else None,
            },
            "by_stratum": by_stratum,
            "by_taxonomy": by_tax,
        }

    # 2. Pairwise inter-rater Cohen's kappa
    inter_rater: dict[str, Any] = {}
    rnames = list(rater_maps.keys())
    for i in range(len(rnames)):
        for j in range(i + 1, len(rnames)):
            r1_name, r2_name = rnames[i], rnames[j]
            m1, m2 = rater_maps[r1_name], rater_maps[r2_name]
            a_lbl, b_lbl = [], []
            for it in primary_items:
                sl = int(it["slot"])
                if sl in m1 and sl in m2:
                    x, y = m1[sl], m2[sl]
                    if not x.skipped and not y.skipped and x.harmful is not None and y.harmful is not None:
                        a_lbl.append(bool(x.harmful))
                        b_lbl.append(bool(y.harmful))
            inter_rater[f"{r1_name}_vs_{r2_name}"] = {
                "n_overlap": len(a_lbl),
                "agreement": float(np.mean([x == y for x, y in zip(a_lbl, b_lbl, strict=True)])) if a_lbl else None,
                "cohens_kappa": cohens_kappa(a_lbl, b_lbl) if a_lbl else None,
            }

    # 3. Consensus human labels (unanimous across all raters who labeled a slot, or single rater if 1 file)
    consensus: dict[str, bool] = {}
    for it in primary_items:
        sl = int(it["slot"])
        votes = [
            bool(rmap[sl].harmful)
            for rmap in rater_maps.values()
            if sl in rmap and not rmap[sl].skipped and rmap[sl].harmful is not None
        ]
        if votes and len(set(votes)) == 1:
            consensus[it["sample_id"]] = votes[0]

    # 4. Optional model accuracy vs Human consensus vs VHD11K judge on exact consensus items
    model_comparison: dict[str, Any] = {}
    cons_sids = sorted(consensus.keys())
    if cons_sids:
        y_human = np.array([consensus[sid] for sid in cons_sids], dtype=bool)
        y_judge = np.array([by_id[sid].harmful for sid in cons_sids], dtype=bool)
        model_comparison["vhd11k_judge"] = {
            "n": len(cons_sids),
            "acc_vs_human": float((y_judge == y_human).mean()),
            "acc_vs_judge": 1.0,
        }
        if probe_path is not None:
            prb = read_predictions(probe_path.expanduser())
            if all(sid in prb and prb[sid].score is not None for sid in cons_sids):
                pred_p = np.array([float(prb[sid].score or 0.0) >= 0.50 for sid in cons_sids], dtype=bool)
                model_comparison["probe"] = {
                    "n": len(cons_sids),
                    "acc_vs_human": float((pred_p == y_human).mean()),
                    "acc_vs_judge": float((pred_p == y_judge).mean()),
                }
        for mname, mpath, z_cut in vlm_specs or []:
            mpreds = read_predictions(mpath.expanduser())
            if all(sid in mpreds and mpreds[sid].score is not None for sid in cons_sids):
                _, z_m, _ = extract_vlm_logits(mpreds, cons_sids)
                pred_m = z_m >= z_cut
                model_comparison[mname] = {
                    "n": len(cons_sids),
                    "z_cut": z_cut,
                    "acc_vs_human": float((pred_m == y_human).mean()),
                    "acc_vs_judge": float((pred_m == y_judge).mean()),
                }

    return {
        "manifest_path": str(manifest_path),
        "n_primary_items": len(primary_items),
        "n_consensus_items": len(cons_sids),
        "per_rater": per_rater,
        "inter_rater": inter_rater,
        "model_vs_human_and_judge": model_comparison,
    }


def _parse_vlm_spec(raw: str) -> tuple[str, Path, float]:
    """Parse NAME=PATH[:Z_CUT] (default z_cut=0.0)."""
    name, sep, rest = raw.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH[:Z_CUT], got {raw!r}")
    if ":" in rest:
        path_str, z_str = rest.rsplit(":", 1)
        return name, Path(path_str), float(z_str)
    return name, Path(rest), 0.0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated label")
    sub = ap.add_subparsers(dest="action", required=True)

    p_prep = sub.add_parser("prepare", help="Create a blind shuffled audit manifest (val-audit or full split)")
    p_prep.add_argument("--mode", choices=("val-audit", "split"), default="val-audit")
    p_prep.add_argument("--dataset", type=Path, required=True)
    p_prep.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    p_prep.add_argument("--split", choices=("train", "val", "test"), default="val")
    p_prep.add_argument("--allow-test", action="store_true", help="Explicitly unlock building a manifest on test")
    p_prep.add_argument("--gate", type=Path, default=Path("results/phase3_val_gate_qwen3_8b.json"))
    p_prep.add_argument("--probe", type=Path, default=Path("results/clip-b16-v2-val.jsonl"))
    p_prep.add_argument("--vlm", type=Path, default=Path("results/qwen3-vl-8b-fp8-v2-val.jsonl"))
    p_prep.add_argument("--in-band-controls", type=int, default=30)
    p_prep.add_argument("--outer-controls", type=int, default=30)
    p_prep.add_argument("--sentinels", type=int, default=12)
    p_prep.add_argument("--limit", type=int, default=None)
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.add_argument("--out", type=Path, default=Path("results/val_audit_120_manifest.json"))

    p_srv = sub.add_parser("serve", help="Launch local blind labeling web UI (pixels only via load_clean)")
    p_srv.add_argument("--dataset", type=Path, required=True)
    p_srv.add_argument("--manifest", type=Path, required=True)
    p_srv.add_argument("--out", type=Path, required=True)
    p_srv.add_argument("--rater", default="vor")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--no-browser", action="store_true")

    p_sc = sub.add_parser("score", help="Compute Cohen's kappa, judge disagreement, and model-vs-human stats")
    p_sc.add_argument("--dataset", type=Path, required=True)
    p_sc.add_argument("--manifest", type=Path, required=True)
    p_sc.add_argument("--raters", type=Path, nargs="+", required=True)
    p_sc.add_argument("--probe", type=Path, default=None)
    p_sc.add_argument(
        "--vlm",
        action="append",
        type=_parse_vlm_spec,
        default=[],
        help="Optional VLM comparison in NAME=PATH[:Z_CUT] format (repeatable)",
    )
    p_sc.add_argument("--json", type=Path, default=None)

    a = ap.parse_args(argv)

    if a.action == "prepare":
        if a.mode == "split" and a.split == "test" and not a.allow_test:
            raise SystemExit("ERROR: splits.v2.json['test'] is sealed. Pass --allow-test only after Phase 4 is locked.")
        if a.mode == "val-audit":
            manifest = build_val_audit_manifest(
                dataset=a.dataset,
                splits_path=a.splits,
                gate_path=a.gate,
                probe_path=a.probe,
                vlm_path=a.vlm,
                n_in_band_control=a.in_band_controls,
                n_outer_control=a.outer_controls,
                n_sentinels=a.sentinels,
                seed=a.seed,
            )
        else:
            manifest = build_split_manifest(
                splits_path=a.splits,
                split_name=a.split,
                n_sentinels=a.sentinels,
                limit=a.limit,
                seed=a.seed,
            )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote blind audit manifest -> {a.out}")
        print("Counts:", manifest["counts"])
        return

    if a.action == "serve":
        serve_blind_ui(
            dataset=a.dataset,
            manifest_path=a.manifest,
            out_path=a.out,
            rater=a.rater,
            host=a.host,
            port=a.port,
            open_browser=not a.no_browser,
        )
        return

    if a.action == "score":
        report = score_human_audit(
            dataset=a.dataset,
            manifest_path=a.manifest,
            rater_paths=a.raters,
            probe_path=a.probe,
            vlm_specs=a.vlm,
        )
        print(json.dumps(report, indent=2))
        if a.json is not None:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote human audit report -> {a.json}")


if __name__ == "__main__":
    main()

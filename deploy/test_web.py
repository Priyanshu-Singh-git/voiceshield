"""Click every test clip in the browser demo (Edge via Playwright); compare browser scores with PyTorch.

  python -m http.server 8603 --directory docs     # other shell
  python deploy/test_web.py http://127.0.0.1:8603/ [--shot assets/figures/demo_screenshot.png]
Writes eval/browser_check.json.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main(url, shot=None):
    meta = json.loads((ROOT / "docs" / "meta.json").read_text())
    thr = meta["threshold"]
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_function("document.getElementById('badge').textContent === 'Ready'", timeout=120000)
        rows, flips, worst, ms = [], 0, 0.0, []
        for i, s in enumerate(meta["samples"]):
            pg.locator("#samples .chip").nth(i).click()
            pg.wait_for_function(f"window.voiceshield.results['{s['file']}'] !== undefined", timeout=120000)
            r = pg.evaluate(f"window.voiceshield.results['{s['file']}']")
            d = abs(r["score"] - s["py_score"]); worst = max(worst, d); ms.append(r["ms"])
            flips += (r["score"] > thr) != (s["py_score"] > thr)
            rows.append({"label": s["label"], "browser": r["score"], "python": s["py_score"], "real": r["real"]})
            print(f"{s['label']:38s} browser {r['score']:+.3f} python {s['py_score']:+.3f} -> {'REAL' if r['real'] else 'AI'}")
        if shot:
            first_fake = next(i for i, s in enumerate(meta["samples"]) if s["attack"] != "-")
            pg.locator("#samples .chip").nth(first_fake).click(); pg.wait_for_timeout(1500)
            pg.evaluate("window.scrollTo(0, 0)"); pg.wait_for_timeout(300)
            box = pg.locator(".grid").bounding_box()
            pg.screenshot(path=str(ROOT / shot), full_page=True, clip={"x": box["x"] - 12, "y": 20, "width": box["width"] + 24, "height": box["y"] + box["height"] - 8})
        b.close()
    ms.sort()
    out = {"clips": len(rows), "max_score_diff": worst, "verdict_flips": flips, "median_ms": ms[len(ms) // 2], "console_errors": errs, "rows": rows}
    (ROOT / "eval" / "browser_check.json").write_text(json.dumps(out, indent=2))
    print(f"clips={len(rows)} max_score_diff={worst:.4f} verdict_flips={flips} median_ms={ms[len(ms) // 2]:.0f} errors={errs[:3]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--shot" else None)

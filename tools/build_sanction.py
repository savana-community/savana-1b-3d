"""Build the structured store for the 2021 revised sanction plan.

Inputs (data/sanction-2021/transcribed/): the five site-number lists from the
"Sites Details" table and the 255-row "ODD SITES" table, transcribed from the
scan by OCR and checked against the scan by eye.
Outputs: data/sanction-2021/sanction.json and plots.csv.
"""
import csv, json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "sanction-2021"
T = D / "transcribed"

CLASSES = {  # key: (frontage_m, depth_m, printed count, printed area sqm)
    "s9x12":  (9.14, 12.19, 485, 54037.05),
    "s9x15":  (9.14, 15.24, 388, 54045.92),
    "s12x18": (12.19, 18.28, 206, 45903.64),
    "s15x24": (15.24, 24.38, 70, 26008.58),
}

def nums(name):
    return [int(x) for x in re.findall(r"\d+", (T / f"{name}.txt").read_text())]

lists = {k: nums(k) for k in [*CLASSES, "odd"]}
odd_table = json.loads((T / "odd_table.json").read_text())
odd_area = {r["site"]: r["area_sqm"] for r in odd_table}
odd_sl = {r["site"]: r["sl"] for r in odd_table}

checks = []
def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})

# coverage: every plot 1..1404 in exactly one class, 1405 = club house
seen = collections.Counter(n for v in lists.values() for n in v)
check("plots 1-1404 each listed exactly once",
      set(seen) == set(range(1, 1405)) and max(seen.values()) == 1)
for k, (f, d, n, area) in CLASSES.items():
    calc = round(f * d * n, 2)
    check(f"{k}: count {n}", len(lists[k]) == n, f"listed {len(lists[k])}")
    check(f"{k}: {f} x {d} x {n} = printed area", abs(calc - area) < 0.01, f"{calc} vs {area}")
check("odd: count 255", len(lists["odd"]) == 255)
check("odd table site numbers == odd list", set(odd_area) == set(lists["odd"]))
odd_sum = round(sum(odd_area.values()), 2)
check("odd table areas sum to printed total 83,646.96", abs(odd_sum - 83646.96) < 0.01,
      f"rows sum to {odd_sum}; every row re-read against the scan, so the printed total is 0.06 higher")

landuse = [
    {"sl": 1, "use": "Residential", "area_sqm": 263642.15, "pct": 53.70},
    {"sl": 2, "use": "Club house", "area_sqm": 5919.86, "pct": 1.21},
    {"sl": 3, "use": "Civic Amenities", "area_sqm": 24557.41, "pct": 5.00},
    {"sl": 4, "use": "Park & Open Space", "area_sqm": 49183.50, "pct": 10.02},
    {"sl": 5, "use": "Road", "area_sqm": 147675.09, "pct": 30.08},
]
parks = [1403.33, 1370.98, 394.20, 2045.72, 1269.27, 1828.71, 1628.12, 620.20, 2839.65, 2773.29,
         4099.70, 1095.52, 7792.82, 8922.23, 9665.30, 85.34, 65.15, 387.06, 896.91]
ca = [16587.13, 4232.65, 1974.08, 1763.55]
res_sum = round(sum(a for *_, a in CLASSES.values()) + 83646.96, 2)
check("residential = sum of plot classes", abs(res_sum - 263642.15) < 0.01, str(res_sum))
check("landuse total = net site area 490,978.01",
      abs(sum(r["area_sqm"] for r in landuse) - 490978.01) < 0.01)
check("parks sum to 49,183.50", abs(sum(parks) - 49183.50) < 0.01, str(round(sum(parks), 2)))
check("CA sum to 24,557.41", abs(sum(ca) - 24557.41) < 0.01, str(round(sum(ca), 2)))

plots = []
for k, (f, d, *_ ) in CLASSES.items():
    for n in lists[k]:
        plots.append({"plot": n, "class": f"{f} x {d}", "frontage_m": f, "depth_m": d,
                      "area_sqm": round(f * d, 2), "area_source": "dimension"})
for n in lists["odd"]:
    plots.append({"plot": n, "class": "odd", "frontage_m": None, "depth_m": None,
                  "area_sqm": odd_area[n], "area_source": f"odd table row {odd_sl[n]}"})
plots.append({"plot": 1405, "class": "club house", "frontage_m": None, "depth_m": None,
              "area_sqm": 5919.86, "area_source": "site details table"})
plots.sort(key=lambda p: p["plot"])

# Nominal drawing scale: page is 2526 x 3493 pt, scanned at 7017 x 9704 px, printed 1:1750.
mm_per_px = 2526 / 72 * 25.4 / 7017
store = {
    "source": {
        "file": "Attachment 1 Revised Sanction Plan.pdf",
        "kind": "single-page scan, no text layer",
        "page_pt": [2526, 3493], "scan_px": [7017, 9704],
        "printed_scale": "1:1750",
        "nominal_m_per_scan_px": round(mm_per_px * 1750 / 1000, 4),
        "nominal_scale_note": "assumes the PDF page is the original sheet size; to be verified against plot dimensions on the drawing",
        "header": "ONLY FOR DEMARCATION AND DEVELOPMENT PURPOSE",
    },
    "project": {
        "title": "Draft modified residential layout plan",
        "survey_numbers": {
            "Bettenahalli": "73/P4, 92, 94, 95, 96, 97(P), 98, 99, 100, 101/12, 101/2, 102, 103, 106/1, 106/2, 106/3, 107, 108/1, 108/2, 108/3, 108/4, 109/1, 109/2, 109/3, 110/1, 111, 112, 113/2, 114, 115, 116, 118, 121/1, 121/2(P), 122",
            "Hegganahalli": "117-149",
        },
        "location": "Kundana Hobli, Devanahalli Taluk, Bengaluru Rural",
    },
    "approval": {
        "type": "Temporary approval (Karnataka Town & Country Planning Act, sec. 17)",
        "authority": "Bengaluru International Airport Area Planning Authority, Devanahalli (Member Secretary)",
        "authority_meeting_date": "2021-08-26",
        "subject_no": "132.77 (handwritten, uncertain)",
        "letter_no": "BIAAPA/TP/01/LAO/20/2019-20 (partly handwritten)",
        "letter_date": "2021-11-24",
        "valid_from": "2021-11-24", "valid_to": "2023-11-23", "signed": "2021-11-24",
        "red_endorsement": "refers to an authority letter dated 2019-12-16 (Kannada handwriting, not fully read)",
        "file_ref": "BCC/BL-36/A-2959/2019-20 (bottom right, low confidence)",
    },
    "area_statement": {
        "total_site_area": {"acres_guntas": "122A-7.0G", "sqm": 494417.79},
        "kharab_area": {"guntas": "34.00G", "sqm": 3439.78},
        "net_site_area": {"acres_guntas": "121A-13G", "sqm": 490978.01},
        "landuse": landuse,
        "residential_plus_club_pct": 54.90,
    },
    "site_classes": [
        {"class": f"{f} x {d}", "frontage_m": f, "depth_m": d, "count": n, "area_sqm": a, "plots": lists[k]}
        for k, (f, d, n, a) in CLASSES.items()
    ] + [
        {"class": "odd", "count": 255, "area_sqm": 83646.96, "plots": lists["odd"]},
        {"class": "club house", "count": 1, "area_sqm": 5919.86, "plots": [1405]},
    ],
    "totals": {"plots": 1405, "area_sqm": 269562.01},
    "odd_sites_table": odd_table,
    "parks": [{"park": f"Park-{i+1}", "area_sqm": a} for i, a in enumerate(parks)],
    "civic_amenities": [{"ca": f"CA-{i+1}", "area_sqm": a} for i, a in enumerate(ca)],
    "legend": ["Site boundary", "Plots", "Survey hissa line", "Roads", "Building line",
               "Nala buffer line", "Parks", "Commercial", "C.A."],
    "transcription_notes": [
        "Odd table row 206 prints site 1170, duplicating row 205; the site list has 1170 and 1171, so row 206 is recorded as 1171.",
        "Odd table areas sum to 83,646.90; the printed total is 83,646.96.",
        "Site-number lists were OCR'd, then patched by reading the scan (dropped lines 1252-1272 in 9.14x12.19 and 1043-1150 in 15.24x24.38, plus single-digit misreads); the coverage check confirms the result.",
        "Not transcribed: the Kannada land-records table at top left (survey numbers, extents, conversion orders), and plot numbers and dimensions on the map itself.",
    ],
    "checks": checks,
    "plots": plots,
}
(D / "sanction.json").write_text(json.dumps(store, indent=1, ensure_ascii=False))
with open(D / "plots.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(plots[0]))
    w.writeheader(); w.writerows(plots)
(D / "sanction-data.js").write_text("window.SANCTION = " + json.dumps(store, ensure_ascii=False) + ";\n")
for c in checks:
    print("PASS" if c["ok"] else "FAIL", c["check"], c["detail"])
print(len(plots), "plots written")

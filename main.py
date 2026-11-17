"""GTA VI Twitch CCU forecast (make.fun gta6-twitch-ccu-forecast-01).

Reads one case per *.in file (Twitch category viewers up to the end of
release day's hour 1, plus the full hourly curves of ~40 past big game
launches as comparables) and writes <id>.out with five quantiles
(0.05/0.25/0.5/0.75/0.95) of the category's concurrent viewers for each of
hours 2..24 of the release day (UTC).

Method
------
The forecast is anchored on hour 1's print p1 and predicts, per hour h,
the log ratio L_h = ln(y_h / p1).  Each past launch gives one observed
ratio curve.  Launches are weighted by how similar they are to the case
at hand, with a Gaussian kernel over four features:

  m     = ln(p1 / baseline)   -- how big the launch jump is vs the
                                 category's quiet pre-launch level
  s     = ln(p1)              -- absolute scale (big launches saturate)
  pre1  = ln(viewers 1h before day start / baseline)
  pre12 = ln(median viewers in the 12h before day start / baseline)

plus a penalty when a launch's unlock pattern (rolling midnight local vs
one simultaneous global unlock) differs from the case's, and zero weight
for launches flagged weak or early-access-contaminated.  The baseline is
the median of the category's points from 14 days to 1 day before release.

The ensemble (MEMBERS below) is averaged in log space; every member is
smoothed over neighbouring hours (window +-1):
  A: kernel-weighted mean of L_h, normal quantiles, bw(m)=2.0, bw(s)=0.7
  B: kernel-weighted mean of L_h, normal quantiles, bw(m)=1.5, bw(s)=1.0,
     spread x0.85
  C: kernel-weighted empirical quantiles of L_h directly (captures the
     right skew of launch curves)
  D: hybrid -- mean centre with empirical-quantile spread around the
     weighted median

Validation
----------
Chosen by leave-one-out cross-validation over the 37 usable comparable
launches in the kit, scoring the shipped program end to end (pinball
loss on the log scale, the contest's metric): 0.465 overall, 0.206 on
rolling-midnight launches, 0.193 on GTA-like ones (rolling, hour-1
viewers > 30k, clean), 0.154 on the largest launches (hour-1 > 300k),
vs 0.930 / 0.363 / 0.352 / 0.226 for the house persistence baseline.
Variants tried and rejected on the same cross-validation: blending
toward persistence or toward the global mean curve when the kernel's
effective sample size is small (worse everywhere), year-decay weights
(worse), per-hour sigma/bias calibration from leave-one-out residuals
(worse), pre-launch ramp-slope and viewers-per-channel features (no
gain), k-nearest neighbours instead of the Gaussian kernel (worse), and
widening quantiles by cross-member disagreement (neutral on rolling
launches).  The plain ensemble above was best on every GTA-relevant
subset, so it ships as-is.

Rolling-midnight console launches are GTA VI's regime: it unlocks at
midnight local, New Zealand first (11:00Z Nov 18), UK at 00:00Z, US
East at 05:00Z, so hour 1 already carries Australasian/Japanese play
plus the UK unlock, and the audience then ramps through the day into
the European evening and US afternoon/evening peak (hours ~16-24 UTC).

Standard library only, no network, deterministic (no randomness).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
HOURS = list(range(1, 24))  # offsets 1..23 == hours 2..24

# Ensemble members: (kernel features with bandwidths, kind, sigma scale).
# kind: "normal-mean" (weighted mean centre, normal quantiles),
# "emp" (weighted empirical quantiles), "hybrid" (mean centre,
# empirical-quantile spread around the weighted median).
MEMBERS = [
    ([("m", 2.0), ("s", 0.7), ("pre1", 1.5), ("pre12", 1.5)],
     "normal-mean", 1.0),
    ([("m", 1.5), ("s", 1.0), ("pre1", 1.5), ("pre12", 1.5)],
     "normal-mean", 0.85),
    ([("m", 1.5), ("s", 1.0), ("pre1", 1.5), ("pre12", 1.5)],
     "emp", 1.0),
    ([("m", 1.5), ("s", 1.0), ("pre1", 1.5), ("pre12", 1.5)],
     "hybrid", 1.0),
]


def parse_time(t):
    return datetime.fromisoformat(str(t).replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, as JavaScript Math.round does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def wmean(vals, ws):
    tot = sum(ws)
    if tot <= 0:
        return None
    return sum(v * w for v, w in zip(vals, ws)) / tot


def wmedian(vals, ws):
    pairs = sorted(zip(vals, ws))
    tot = sum(ws)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def wquantile(vals, ws, q):
    pairs = sorted(zip(vals, ws))
    tot = sum(ws)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * tot:
            return v
    return pairs[-1][0]


def wstd(vals, ws, mu):
    tot = sum(ws)
    if tot <= 0:
        return 0.5
    return math.sqrt(max(sum(w * (v - mu) ** 2 for v, w in zip(vals, ws)) / tot, 0.0))


def kernel_weight(tgt, cp, feats, roll_penalty=1.0):
    d2 = 0.0
    for feat, bw in feats:
        a = cp.get(feat)
        b = tgt.get(feat)
        if a is None or b is None:
            continue
        d2 += ((a - b) / bw) ** 2
    if roll_penalty and tgt.get("roll") is not None and cp.get("roll") is not None:
        if tgt["roll"] != cp["roll"]:
            d2 += roll_penalty ** 2
    return math.exp(-0.5 * d2)


def rolling_from_text(text):
    """1 = rolling midnight-local unlock, 0 = one simultaneous global
    unlock, None = cannot tell from the text."""
    t = (text or "").lower()
    if "rolling" in t or "midnight local" in t:
        return 1
    if "single global" in t or "simultaneous" in t:
        return 0
    return None


def median_or_none(vals):
    vals = [v for v in vals if v is not None and v > 0]
    return statistics.median(vals) if vals else None


def build_launch(hourly, unlock_text, target_name, name):
    """hourly: {offset_hours_from_day_start: viewers}.  Returns a feature
    dict, or None when the launch cannot anchor a ratio curve."""
    h1 = hourly.get(0)
    if h1 is None or h1 <= 0:
        return None
    if name and target_name and name.strip().lower() == target_name.strip().lower():
        return None  # the target's own category history, not a launch
    base_pts = [v for off, v in hourly.items() if -336 <= off < -24 and v > 0]
    base = statistics.median(base_pts) if base_pts else 1.0
    base = max(base, 1.0)
    pre1 = hourly.get(-1)
    pre12 = median_or_none([hourly.get(off) for off in range(-12, 0)])
    return {
        "name": name,
        "hourly": hourly,
        "h1": float(h1),
        "B": base,
        "m": math.log(h1 / base),
        "s": math.log(h1),
        "pre1": math.log(max(pre1, 1) / base) if pre1 is not None else None,
        "pre12": math.log(max(pre12, 1) / base) if pre12 is not None else None,
        "roll": rolling_from_text(unlock_text),
        "weak": ("tier: weak" in (unlock_text or "").lower()),
        "ea": ("early-access-contaminated" in (unlock_text or "").lower()),
    }


def target_features(case):
    day_start = parse_time(case["day_start"])
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))

    # Pre-launch points: history_hourly preferred, else history.
    pts = []
    hh = case.get("history_hourly") or []
    if hh:
        for p in hh:
            t = parse_time(p.get("from") or p.get("at"))
            pts.append((t, float(p.get("viewers", 0) or 0)))
    else:
        for p in case.get("history") or []:
            t = parse_time(p.get("at") or p.get("from"))
            pts.append((t, float(p.get("viewers", 0) or 0)))

    base_pts = [v for t, v in pts
                if day_start - timedelta(days=14) <= t < day_start - timedelta(hours=24)
                and v > 0]
    base = max(statistics.median(base_pts) if base_pts else 1.0, 1.0)
    before = [(t, v) for t, v in pts if t < day_start]
    pre1 = None
    if before:
        pre1 = max(before, key=lambda tv: tv[0])[1]
    pre12 = median_or_none([v for t, v in pts
                            if day_start - timedelta(hours=12) <= t < day_start])
    game = case.get("game") or {}
    return {
        "h1": p1,
        "B": base,
        "m": math.log(p1 / base),
        "s": math.log(p1),
        "pre1": math.log(max(pre1, 1) / base) if pre1 is not None else None,
        "pre12": math.log(max(pre12, 1) / base) if pre12 is not None else None,
        "roll": rolling_from_text(case.get("notes") or ""),
        "name": game.get("name") if isinstance(game, dict) else None,
    }


def member_forecast(tgt, comps, feats, kind, sig_scale=1.0):
    """One ensemble member: per-hour log-quantiles (23 x 5)."""
    ws = [kernel_weight(tgt, cp, feats) for cp in comps]
    logq = []
    for h in HOURS:
        vals, wts = [], []
        for cp, w in zip(comps, ws):
            y = cp["hourly"].get(h)
            if y is None or y <= 0:
                continue
            vals.append(math.log(y / cp["h1"]))
            wts.append(w)
        if not vals:
            logq.append(None)
            continue
        if kind == "emp":
            qs = [wquantile(vals, wts, tau) for tau in LEVELS]
        else:
            mu = wmean(vals, wts)
            if kind == "hybrid":
                med = wmedian(vals, wts)
                qs = [mu + (wquantile(vals, wts, tau) - med) for tau in LEVELS]
            else:  # normal-mean
                sig = min(max(wstd(vals, wts, mu), 0.30), 0.9) * sig_scale
                qs = [mu + z * sig for z in Z]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        logq.append(qs)
    # Fall back to the house baseline shape wherever a member has no data.
    for i, qs in enumerate(logq):
        if qs is None:
            h = HOURS[i]
            logq[i] = [z * 0.15 * math.sqrt(h) for z in Z]
    # Smooth each quantile level across neighbouring hours.
    out = []
    for i in range(len(logq)):
        seg = logq[max(0, i - 1):min(len(logq), i + 2)]
        out.append([sum(row[j] for row in seg) / len(seg) for j in range(5)])
    for qs in out:
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
    return out


def baseline_logq():
    return [[z * 0.15 * math.sqrt(h) for z in Z] for h in HOURS]


def forecast_case(case):
    tgt = target_features(case)
    game_name = tgt.get("name")
    comps = []
    for comp in case.get("comparables") or []:
        hourly = {}
        for p in comp.get("hourly") or []:
            try:
                off = int((parse_time(p["from"]) - parse_time(comp["day_start"]))
                          .total_seconds() // 3600)
            except Exception:
                continue
            hourly[off] = float(p.get("viewers", 0) or 0)
        launch = build_launch(hourly, comp.get("unlock") or "",
                              game_name, comp.get("name"))
        if launch is not None:
            comps.append(launch)
    # Prefer clean comparables; fall back to all of them if none are clean.
    clean = [c for c in comps if not c["weak"] and not c["ea"]]
    if clean:
        comps = clean
    if not comps:
        rows = baseline_logq()
        return {"hours": [{"hour": h + 1, "q": [round2(tgt["h1"] * math.exp(q))
                                                for q in row]}
                          for h, row in zip(HOURS, rows)]}

    members = [member_forecast(tgt, comps, feats, kind, sig_scale)
               for feats, kind, sig_scale in MEMBERS]
    hours = []
    for idx, h in enumerate(HOURS):
        qs = []
        for j in range(5):
            avg = sum(m[idx][j] for m in members) / len(members)
            qs.append(round2(min(max(tgt["h1"] * math.exp(avg), 0.01), 1e9)))
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h + 1, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

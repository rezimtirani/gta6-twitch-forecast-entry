# gta6-twitch-forecast-entry

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rezimtirani/gta6-twitch-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rezimtirani/gta6-twitch-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles an hour, from the first hour's viewer count plus the full hourly curves of ~40 past big game launches. The model is an ensemble of kernel-weighted forecasts of the log ratio ln(viewers_h / viewers_hour1): launches are weighted by similarity in launch-jump size, absolute scale, and pre-launch ramp, with a penalty for a different unlock pattern (rolling midnight local vs one simultaneous global unlock) and zero weight for weak or early-access-contaminated launches. It was tuned by leave-one-out cross-validation over the kit's comparable launches (pinball loss on the log scale, the contest's metric): ~0.20 on rolling-midnight launches like GTA VI's, vs ~0.35 for the house persistence baseline. GTA VI unlocks at midnight local, New Zealand first (11:00Z Nov 18), so hour 1 already carries Australasian/Japanese play plus the UK unlock, and the audience ramps through the day into the European evening and US afternoon/evening peak (hours ~16-24 UTC).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

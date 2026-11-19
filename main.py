"""Barcelona Urban Flood Prediction & Transport Monitoring Platform — entry point.

Usage
-----
  python main.py setup      # first run: network + weather + synthetic sensors + train + evaluate
  python main.py serve      # start the dashboard on http://127.0.0.1:8050 (default command)
  python main.py update     # refresh weather (history + live forecast) and sensor data
  python main.py network    # (re)build transport network layers and monitored segments
  python main.py sensors    # (re)generate the synthetic sensor series
  python main.py train      # train the probabilistic models
  python main.py evaluate   # evaluate on the test period -> reports/
  python main.py forecast   # print the current forecast summary

Decision-support prototype: final operational decisions remain with
authorised TMB personnel.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger("main")


def cmd_network(args):
    from geospatial.network import build_network
    print(build_network(offline_streets=getattr(args, "offline", False)))


def cmd_update(args):
    from data_collection import weather_service
    from data_collection.sensors_synthetic import generate
    print("weather history:", weather_service.update_history())
    print("live forecast  :", weather_service.update_forecast())
    from data_collection import meteo_stations, transport_tmb
    meteo_stations.fetch_meteocat()
    meteo_stations.fetch_aemet()
    transport_tmb.download_gtfs()
    generate()
    print("synthetic sensors updated")


def cmd_sensors(args):
    from data_collection.sensors_synthetic import generate
    df = generate()
    print(f"{len(df):,} synthetic observations written")


def cmd_train(args):
    from models.train import train
    print("model version:", train())


def cmd_evaluate(args):
    from models.evaluate import evaluate
    evaluate(compare_models=not args.quick)
    print("report written to reports/model_report.md")


def cmd_forecast(args):
    from models.predict import forecast
    r = forecast(args.issue)
    print(f"issued {r['issued_at']} ({r['mode']}), weather: {r['weather_forecast']['source']}")
    for s in r["sensors"]:
        hz = {h["h"]: h for h in s["horizons"]}
        p = " ".join(f"+{h}h:{hz[h]['p_exceed_max']['critical']:.2f}" for h in (1, 3, 6, 24) if h in hz)
        print(f"{s['sensor_id']:7s} now {s['current']['level']!s:>6} cm | P(max>crit) {p} | {s['name']}")


def cmd_setup(args):
    from data_storage import db
    db.init_db()
    log.info("1/6 transport network (OpenStreetMap)")
    cmd_network(args)
    log.info("2/6 weather history + archived forecasts (Open-Meteo)")
    from data_collection import weather_service
    print(weather_service.update_history())
    log.info("3/6 live weather forecast")
    print(weather_service.update_forecast())
    log.info("4/6 synthetic sensors")
    cmd_sensors(args)
    log.info("5/6 model training (this takes ~10-20 min)")
    cmd_train(args)
    log.info("6/6 evaluation")
    args.quick = getattr(args, "quick", False)
    cmd_evaluate(args)
    print("\nSetup complete. Start the dashboard with:  python main.py serve")


def cmd_serve(args):
    import uvicorn
    from api.server import app, start_scheduler
    from config.settings import settings
    cfg = settings()["api"]
    if not args.no_scheduler:
        start_scheduler()
    print(f"\n  Dashboard: http://{cfg['host']}:{cfg['port']}\n")
    uvicorn.run(app, host=cfg["host"], port=args.port or cfg["port"], log_level="warning")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="BCN Flood & Transport Monitoring Platform")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("setup"); s.add_argument("--quick", action="store_true", help="skip learner comparison")
    s.add_argument("--offline", action="store_true", help="use cached OSM street data only")
    sub.add_parser("update")
    n = sub.add_parser("network"); n.add_argument("--offline", action="store_true")
    sub.add_parser("sensors")
    sub.add_parser("train")
    e = sub.add_parser("evaluate"); e.add_argument("--quick", action="store_true", help="skip learner comparison")
    f = sub.add_parser("forecast"); f.add_argument("--issue", default=None, help="ISO time (replay)")
    sv = sub.add_parser("serve"); sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--no-scheduler", action="store_true")
    args = p.parse_args(argv)
    if args.cmd is None:
        args = p.parse_args(["serve"] + (argv or sys.argv[1:]))
    {"setup": cmd_setup, "update": cmd_update, "network": cmd_network, "sensors": cmd_sensors,
     "train": cmd_train, "evaluate": cmd_evaluate, "forecast": cmd_forecast, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()

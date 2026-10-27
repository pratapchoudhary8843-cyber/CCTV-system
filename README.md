# CCTV-system
# Smart CCTV

**An AI-Based Intelligent Video Surveillance System**

Turn an ordinary camera into a security system that understands what it sees. Smart CCTV detects and tracks people and objects, raises an alert when someone enters a restricted zone, counts crowds, records short evidence clips, sends notifications and shows everything on a live web dashboard with analytics.

*Mini project for B.Sc. Data Science (Second Year), St. Wilfred's College of Computer Sciences, Mumbai University, 2026-27.*
*By **Pratap Choudhary**, under the guidance of **Prof. Firoza Mirza**.*

![Dashboard](docs/dashboard.png)

## Features

| Feature | What it does |
| --- | --- |
| Person and object detection | YOLOv8 finds people and common objects in every frame and draws labelled boxes |
| Person tracking | Each person gets an ID, so the dashboard can count *different* people today |
| Intrusion detection | Draw a restricted zone on the live video; a person standing inside it raises an intrusion event |
| Crowd detection | When the number of people reaches a limit (5 by default) a crowd event is raised |
| Watched objects | Backpack, suitcase and knife raise an object event (the list can be changed) |
| Event recording | Every event saves a snapshot and a clip: 3 seconds before to 5 seconds after |
| Notifications | Telegram (with the snapshot) or any webhook (Slack, Discord, n8n and more), with a cooldown between repeats |
| Analytics dashboard | Live counts, events by hour, average people by hour, busiest hour, event history |
| Data storage | Events and per-minute people counts are kept in a local SQLite database |

## Quick start

You need **Python 3.9 or newer** and a webcam (or any video file).

```bash
pip install -r requirements.txt
python app.py
```

Your browser opens **http://localhost:8000**. If the camera does not start by itself, click **Use this computer's camera** and choose **Allow**.

The first start downloads the YOLO model (about 6 MB), so allow a minute.

## Choosing the camera

By default (`SOURCE=auto`) the server tries camera 0 and, if that fails, the dashboard offers the browser camera. Set `SOURCE` before running `python app.py` to choose directly.

| Camera | Value |
| --- | --- |
| Automatic (default) | `auto` |
| Browser camera only | `browser` |
| USB or laptop camera number | `0`, `1`, `2` |
| IP camera | `rtsp://user:password@192.168.1.10/stream1` |
| Video file (for testing) | `C:\videos\test.mp4` |

How to set it:

```bash
# Windows (Command Prompt)
set SOURCE=1
python app.py

# Windows (PowerShell)
$env:SOURCE = "1"; python app.py

# macOS and Linux
SOURCE=1 python app.py
```

Not sure which camera number works? Run `python test_camera.py`.

## Using the dashboard

1. **Draw the restricted zone:** click *Edit restricted zone*, click 3 or more corner points on the video, then *Save zone*. It is remembered after a restart.
2. **Watch the numbers:** intrusions, crowd alerts, people in view now, different people today, watched objects and the busiest hour.
3. **Review events:** the table shows a snapshot, type, time and a link to the clip. Colours: red = intrusion, purple = crowd, amber = watched object, green = person.

![Charts and events](docs/charts-and-events.png)

![Event history](docs/event-history.png)

## Settings

Set these as environment variables before starting.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOURCE` | `auto` | Camera source (see above) |
| `MODEL` | `yolov8n.pt` | Detection model. `yolov8s.pt` or `yolov8m.pt` find more people but are slower |
| `IMGSZ` | `960` | Detection size. Higher finds smaller or more distant people; lower is faster |
| `CONF` | `0.35` | Minimum detection confidence |
| `CROWD_LIMIT` | `5` | People needed for a crowd alert |
| `WATCH_OBJECTS` | `backpack,suitcase,knife` | Objects that raise an object event |
| `NOTIFY` | `intrusion,object,crowd` | Event types that send notifications |
| `COOLDOWN` | `30` | Seconds before the same kind of event is saved and alerted again |
| `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` | none | Telegram alerts (create a bot with @BotFather) |
| `WEBHOOK_URL` | none | Send alerts as JSON to this address |
| `PORT` | `8000` | Server port |

## How it works

```
Camera (browser / USB / RTSP / file)
        |
        v
YOLOv8 detection + tracking  ->  rules: intrusion, crowd, watched objects
        |                                   |
        v                                   v
live annotated video                 event: snapshot + clip + database row + alert
        |                                   |
        +---------------> FastAPI <---------+
                              |
                              v
                     web dashboard (charts, events)
```

## Project structure

```
smart-cctv/
  app.py              backend: detection, rules, recording, alerts, API
  static/index.html   dashboard page
  requirements.txt    Python packages
  test_camera.py      finds a working camera number
  docs/               screenshots used in this README
  data/               created when you run it: cctv.db, clips/, snaps/, zone.json
```

## API

The dashboard uses these automatically. Interactive documentation is at http://localhost:8000/docs.

| Route | Purpose |
| --- | --- |
| `POST /api/frame` | Send one JPEG picture from the browser camera for detection |
| `GET /video` | Live annotated video (MJPEG stream) |
| `GET /api/live` | Current counts, people today, status, frame rate, errors |
| `GET /api/events` | Latest events with links to snapshots and clips |
| `GET /api/analytics` | Events by hour, average people by hour, today's totals, busiest hour |
| `GET`, `POST /api/zone` | Read or save the restricted zone |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Red banner or "Internal Server Error" | The banner and the terminal show the real reason. Run `pip install -r requirements.txt` again and restart |
| Camera does not open | Click Allow in the browser prompt. Close Zoom, Teams and other tabs using the camera. Windows: Settings > Privacy > Camera. Try `SOURCE=1` or run `python test_camera.py` |
| No camera (WSL, Docker, cloud server) | Use `SOURCE` with an RTSP address or a video file |
| "Loading AI model" never ends | The first run needs internet to download the model |
| Video is choppy | Set `IMGSZ=640`, use `yolov8n.pt`, or use a computer with a GPU |
| Clip will not play in the browser | Download it and open it in a video player; the codec depends on your computer |
| Misses people in a big crowd | Use `MODEL=yolov8s.pt` or `yolov8m.pt`, raise `IMGSZ` to 1280, mount the camera higher |

## Limitations

- One camera and one restricted zone at a time.
- Detection is not perfect: people can be missed in poor light, heavy crowds or when hidden.
- With the browser camera, detection runs only while the dashboard tab is open. For 24-hour use, run an RTSP or USB camera with `SOURCE`.
- The dashboard has no login. Keep it on your own network or behind a VPN or HTTPS reverse proxy.
- The count of different people resets at midnight or when the program restarts.

## Privacy and responsible use

Video of people is personal data. Check the local law before installing cameras, put up a notice where people may be recorded, never point the camera at private places, keep the `data` folder private and delete old clips regularly. Do not rely on this system as the only safety or security measure.

## Future work

Several cameras and zones, login and HTTPS, behaviour detection (loitering, falling, fighting), mobile push notifications, automatic deletion of old clips, edge devices such as Jetson or Raspberry Pi, and daily or weekly reports.

## Built with

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org), [FastAPI](https://fastapi.tiangolo.com), [SQLite](https://www.sqlite.org) and [Chart.js](https://www.chartjs.org).

import os
import cv2
import time
import pickle
import requests
import threading
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime, time as dtime

try:
    from anti_spoof import LivenessChecker
    checker = LivenessChecker()
except Exception as e:
    print("Anti-spoof model could not be loaded:", e)
    print("Install torch and put the Silent-Face-Anti-Spoofing folder next to this file.")
    raise SystemExit(1)

CUTOFF = dtime(23 , 30)
API_URL = "https://umrao.pythonanywhere.com/api/mark"
API_KEY = "Pkumrao@9956"

MATCH_THRESHOLD = 0.40
MARGIN = 0.08
SHOW_DISTANCE = True

WINDOW = 3
MIN_FRAMES = 2
REAL_THRESHOLD = 0.65
REAL_FRACTION = 0.65
STABLE_FRACTION = 0.80
TRACK_TIMEOUT = 1.5

with open("encodings/face_model.pkl", "rb") as file:
    data = pickle.load(file)

known_encodings = data["encodings"]
known_names = data["names"]
known_rollnos = data["rollnos"]

student_index = {}

for i, r in enumerate(known_rollnos):
    student_index.setdefault(str(r), []).append(i)

name_of = dict(zip(map(str, known_rollnos), known_names))

attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Roll No", "Name", "Date", "Time", "Status"])
    df.to_csv(attendance_file, index=False)

today = datetime.now().strftime("%d-%m-%Y")

df = pd.read_csv(attendance_file)
today_attendance = df[df["Date"] == today]
marked_students = set(today_attendance["Roll No"].astype(str))

absent_done = False
tracks = []


def sync_to_server(payload):
    try:
        requests.post(API_URL, json=payload, timeout=5)
    except Exception as e:
        print("Sync failed:", e)


def save_record(rollno, name, status, now):
    new_record = pd.DataFrame([{
        "Roll No": rollno,
        "Name": name,
        "Date": now.strftime("%d-%m-%Y"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": status   
    }])

    new_record.to_csv(attendance_file, mode="a", header=False, index=False)

    threading.Thread(
        target=sync_to_server,
        args=({
            "key": API_KEY,
            "rollno": rollno,
            "name": name,
            "status": status,
            "date": now.strftime("%d-%m-%Y"),
            "time": now.strftime("%H:%M:%S")
        },),
        daemon=True
    ).start()


def mark_attendance(rollno, name):
    rollno = str(rollno)

    if rollno in marked_students:
        return

    now = datetime.now()

    if now.time() > CUTOFF:
        return

    marked_students.add(rollno)
    save_record(rollno, name, "Present", now)

    print(f"Attendance marked (live face verified): {rollno} - {name}")


def mark_absent_after_cutoff():
    global absent_done

    if absent_done:
        return

    now = datetime.now()

    if now.time() <= CUTOFF:
        return

    absent_done = True

    for roll, name in set(zip(map(str, known_rollnos), known_names)):
        if roll in marked_students:
            continue

        marked_students.add(roll)
        save_record(roll, name, "Absent", now)

    print("Cutoff passed: absent marked for remaining students")


def identify(face_encoding):
    distances = face_recognition.face_distance(known_encodings, face_encoding)

    best = {}

    for roll, idx in student_index.items():
        best[roll] = float(np.min(distances[idx]))

    ranked = sorted(best.items(), key=lambda x: x[1])

    roll, distance = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 1.0

    if distance <= MATCH_THRESHOLD and (second - distance) >= MARGIN:
        return roll, distance

    return None, distance


def near(a, b):
    ax, ay = (a[1] + a[3]) / 2, (a[0] + a[2]) / 2
    bx, by = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
    size = max(b[1] - b[3], b[2] - b[0])

    return abs(ax - bx) <= size * 0.6 and abs(ay - by) <= size * 0.6


def get_track(box):
    now = time.time()

    for track in tracks:
        if near(box, track["box"]):
            track["box"] = box
            track["seen"] = now
            return track

    track = {"box": box, "scores": [], "rolls": [], "seen": now}
    tracks.append(track)

    return track


def verdict_for(track):
    scores = track["scores"]

    if len(scores) < MIN_FRAMES:
        return "CHECKING"

    real_fraction = sum(s >= REAL_THRESHOLD for s in scores) / len(scores)

    return "REAL" if real_fraction >= REAL_FRACTION else "FAKE"


def stable_roll(track):
    rolls = [r for r in track["rolls"] if r]

    if len(track["rolls"]) < MIN_FRAMES or not rolls:
        return None

    roll = max(set(rolls), key=rolls.count)

    if rolls.count(roll) / len(track["rolls"]) >= STABLE_FRACTION:
        return roll

    return None


video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera started with anti-spoof model.")
print("Look at the camera for 2 to 3 seconds. Photos and phone screens are blocked.")
print("Press Q to quit.")

latest_frame = None
results = []
running = True
size_printed = False


def camera_reader():
    global latest_frame, running

    while running:
        ret, f = video.read()

        if not ret:
            print("Camera error.")
            running = False
            break

        latest_frame = f


def face_worker():
    global results, tracks, size_printed

    while running:
        f = latest_frame

        if f is None:
            time.sleep(0.01)
            continue

        if not size_printed:
            print("Camera frame size:", f.shape[1], "x", f.shape[0])
            size_printed = True

        small_frame = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        now = time.time()
        tracks = [t for t in tracks if now - t["seen"] < TRACK_TIMEOUT]

        new_results = []

        if face_locations:
            rgb_full = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            full_locations = [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in face_locations]
            face_encodings = face_recognition.face_encodings(rgb_full, full_locations)
            detections = checker.detect(f)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                t, r, b, l = face_location
                bbox = [l * 2, t * 2, (r - l) * 2, (b - t) * 2]

                roll, distance = identify(face_encoding)
                score = checker.real_score(f, bbox, detections)

                track = get_track(face_location)
                track["scores"] = (track["scores"] + [score])[-WINDOW:]
                track["rolls"] = (track["rolls"] + [roll])[-WINDOW:]

                verdict = verdict_for(track)
                stable = stable_roll(track)

                if verdict == "REAL" and stable is not None:
                    mark_attendance(stable, name_of[stable])

                shown = stable or roll
                name = name_of[shown] if shown else "Unknown"

                new_results.append((face_location, name, shown or "", distance, score, verdict))

        results = new_results
        time.sleep(0.01)


threading.Thread(target=camera_reader, daemon=True).start()
threading.Thread(target=face_worker, daemon=True).start()

while running:
    mark_absent_after_cutoff()

    frame = latest_frame

    if frame is None:
        time.sleep(0.01)
        continue

    display = cv2.flip(frame, 1)
    w = frame.shape[1]

    for (top, right, bottom, left), name, rollno, distance, score, verdict in list(results):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        left, right = w - right, w - left

        if verdict == "FAKE":
            state = "FAKE - PROXY BLOCKED"
            color = (0, 0, 255)
        elif verdict == "CHECKING":
            state = "Checking liveness..."
            color = (0, 165, 255)
        elif not rollno:
            state = "Unknown"
            color = (0, 0, 255)
        elif rollno in marked_students:
            state = "Marked (live face)"
            color = (0, 255, 0)
        else:
            state = "Live face verified"
            color = (0, 255, 0)

        cv2.rectangle(display, (left, top), (right, bottom), color, 2)

        label = f"{rollno} {name}"

        if SHOW_DISTANCE:
            label += f" real={score:.2f}"

        cv2.rectangle(display, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(display, label, (left + 6, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.putText(display, state, (left, max(top - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imshow("AI College Attendance", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break


time.sleep(0.2)
video.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")
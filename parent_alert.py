import os
import math
import smtplib
import pandas as pd
from datetime import datetime
from email.message import EmailMessage

SENDER_EMAIL = "prateekumrao2008@gmail.com"
APP_PASSWORD = "hrlz jvnw znwf ihwx"
COLLEGE_NAME = "PSIT Kanpur"
LIMIT = 75
DRY_RUN = False

ATTENDANCE_FILE = "attendance.csv"
PARENTS_FILE = "parents.csv"
LOG_FILE = "alerts_sent.csv"


def load_log():
    if not os.path.exists(LOG_FILE):
        return set()

    log = pd.read_csv(LOG_FILE, dtype=str)

    return set(zip(log["Date"], log["Roll No"]))


def save_log(date, roll):
    is_new = not os.path.exists(LOG_FILE)

    pd.DataFrame([{"Date": date, "Roll No": roll}]).to_csv(
        LOG_FILE, mode="a", header=is_new, index=False
    )


def send_email(to, subject, body):
    if DRY_RUN:
        print("---- DRY RUN (no email sent) ----")
        print("To:", to)
        print("Subject:", subject)
        print(body)
        print()
        return False

    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
                server.login(SENDER_EMAIL, APP_PASSWORD.replace(" ", ""))
                server.send_message(msg)
        except (smtplib.SMTPServerDisconnected, OSError):
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
                server.starttls()
                server.login(SENDER_EMAIL, APP_PASSWORD.replace(" ", ""))
                server.send_message(msg)

        return True

    except Exception as e:
        print("Email failed:", to, e)
        return False


def student_stats(df, roll):
    total = df["Date"].nunique()
    present_rows = df[(df["Roll No"] == roll) & (df["Status"].str.strip().str.lower() == "present")]
    present = present_rows["Date"].nunique()
    percent = round(present / total * 100, 1) if total else 0

    return present, total, percent


def build_message(parent, student, roll, date, present, total, percent):
    lines = [
        f"Dear {parent},",
        "",
        f"This is to inform you that {student} (Roll No {roll}) was marked ABSENT on {date}.",
        "",
        f"Current attendance: {percent}% ({present} of {total} days).",
    ]

    if percent < LIMIT:
        needed = math.ceil((LIMIT / 100 * total - present) / (1 - LIMIT / 100))
        lines += [
            "",
            f"Warning: this is below the required {LIMIT}%.",
            f"{student} must attend the next {needed} classes without any absence to reach {LIMIT}%.",
        ]

    lines += [
        "",
        "If you think this is a mistake, please contact the class teacher.",
        "",
        f"- {COLLEGE_NAME} AI Attendance System",
    ]

    return "\n".join(lines)


def send_alerts():
    if not os.path.exists(ATTENDANCE_FILE) or not os.path.exists(PARENTS_FILE):
        print("attendance.csv or parents.csv not found.")
        return

    df = pd.read_csv(ATTENDANCE_FILE, dtype=str, index_col=False)
    parents = pd.read_csv(PARENTS_FILE, dtype=str)

    today = datetime.now().strftime("%d-%m-%Y")
    today_rows = df[df["Date"] == today]

    if today_rows.empty:
        print("No attendance taken today, no alerts sent.")
        return

    log = load_log()
    sent = 0

    for _, p in parents.iterrows():
        roll = str(p["Roll No"]).strip()

        if (today, roll) in log:
            continue

        record = today_rows[today_rows["Roll No"] == roll]
        absent_today = record.empty or record.iloc[-1]["Status"].strip().lower() != "present"

        if not absent_today:
            continue

        present, total, percent = student_stats(df, roll)

        body = build_message(p["Parent Name"], p["Student Name"], roll, today, present, total, percent)
        subject = f"Attendance Alert: {p['Student Name']} was absent on {today}"

        if send_email(p["Parent Email"].strip(), subject, body):
            save_log(today, roll)
            sent += 1

    print(f"Parent alerts sent: {sent}")


if __name__ == "__main__":
    send_alerts()

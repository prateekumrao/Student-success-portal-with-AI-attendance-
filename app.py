
import os
import csv
import pickle
import sqlite3
import joblib
import pandas as pd

from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    session, jsonify, send_from_directory,
    flash, url_for
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ==========================================
# FLASK CONFIGURATION
# ==========================================

BASE = os.path.dirname(os.path.abspath(__file__))

ATT_FILE = os.path.join(BASE, "attendance.csv")
FACE_MODEL = os.path.join(BASE, "encodings", "face_model.pkl")
PRED_MODEL = os.path.join(BASE, "model.pkl")
NOTES_DIR = os.path.join(BASE, "notes")
DB_FILE = os.path.join(BASE, "teachers.db")

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key-before-deployment"
)


# ==========================================
# DATABASE FUNCTIONS
# ==========================================

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_teacher_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            active INTEGER DEFAULT 1
        )
    """)

    conn.commit()
    conn.close()


def create_initial_admin():

    admin_id = os.environ.get("ADMIN_ID")
    password = os.environ.get("ADMIN_PASSWORD")

    if not admin_id or not password:
        print("ADMIN_ID or ADMIN_PASSWORD not configured.")
        return

    conn = get_db()

    password_hash = generate_password_hash(password)

    conn.execute("""
        INSERT INTO teachers
        (teacher_id, name, password_hash, active)
        VALUES (?, ?, ?, 1)

        ON CONFLICT(teacher_id)
        DO UPDATE SET
            name = excluded.name,
            password_hash = excluded.password_hash,
            active = 1
    """, (admin_id, "Administrator", password_hash))

    conn.commit()
    conn.close()

    print("Administrator account created or password updated.")


# ==========================================
# STUDENT FUNCTIONS
# ==========================================

def get_students():

    with open(FACE_MODEL, "rb") as f:
        data = pickle.load(f)

    return dict(zip(data["rollnos"], data["names"]))


def load_attendance():

    if not os.path.exists(ATT_FILE):
        return pd.DataFrame(
            columns=["Roll No", "Name", "Date", "Time", "Status"]
        )

    df = pd.read_csv(ATT_FILE, dtype=str, index_col=False)

    return df[
        ["Roll No", "Name", "Date", "Time", "Status"]
    ].dropna()


def student_stats(roll):

    df = load_attendance()

    total = df["Date"].nunique()

    allrows = df[df["Roll No"] == str(roll)]

    present = allrows[
        allrows["Status"] == "Present"
    ]["Date"].nunique()

    pct = round(present / total * 100, 1) if total else 0

    return {
        "total": total,
        "present": present,
        "absent": total - present,
        "percent": pct,
        "records": allrows.iloc[::-1].to_dict("records")
    }


# ==========================================
# LOGIN DECORATORS
# ==========================================

def login_required(f):

    @wraps(f)
    def wrap(*args, **kwargs):

        if "roll" not in session:
            flash("Please login first.")
            return redirect(url_for("login"))

        return f(*args, **kwargs)

    return wrap


def teacher_required(f):

    @wraps(f)
    def wrap(*args, **kwargs):

        if "teacher_id" not in session:
            flash("Please login as a teacher.")
            return redirect(url_for("teacher_login"))

        return f(*args, **kwargs)

    return wrap


def admin_required(f):

    @wraps(f)
    def wrap(*args, **kwargs):

        if "teacher_id" not in session:
            flash("Please login first.")
            return redirect(url_for("teacher_login"))

        if session["teacher_id"] != "ADMIN":
            flash("Administrator access required.")
            return redirect(url_for("teacher"))

        return f(*args, **kwargs)

    return wrap


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    df = load_attendance()

    today = datetime.now().strftime("%d-%m-%Y")

    return render_template(
        "home.html",
        today_count=int((df["Date"] == today).sum()),
        total_students=len(get_students())
    )


# ==========================================
# STUDENT LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        name = request.form["name"].strip().lower()

        roll = request.form["rollno"].strip()

        students = get_students()

        if roll in students and students[roll].strip().lower() == name:

            session.clear()

            session["roll"] = roll
            session["name"] = students[roll]

            return redirect(url_for("attendance"))

        flash("Invalid name or roll number.")

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))


# ==========================================
# STUDENT ATTENDANCE
# ==========================================

@app.route("/attendance")
@login_required
def attendance():

    return render_template(
        "attendance.html",
        s=student_stats(session["roll"])
    )


# ==========================================
# STUDENT NOTES
# ==========================================

@app.route("/notes")
@login_required
def notes():

    files = (
        sorted(os.listdir(NOTES_DIR))
        if os.path.isdir(NOTES_DIR)
        else []
    )

    return render_template("notes.html", files=files)


@app.route("/notes/<path:fname>")
@login_required
def download_note(fname):

    return send_from_directory(
        NOTES_DIR,
        fname,
        as_attachment=True
    )


# ==========================================
# STUDENT PREDICTION
# ==========================================

@app.route("/predict")
@login_required
def predict_page():

    return render_template(
        "predict.html",
        att=student_stats(session["roll"])["percent"]
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():

    d = request.get_json()

    X = pd.DataFrame([{
        "Study_Time_Hours": float(d["study"]),
        "Attendance_Percentage": float(d["attendance"]),
        "Previous_Exam_Score": float(d["score"])
    }])

    model = joblib.load(PRED_MODEL)

    p = float(model.predict_proba(X)[0][1])

    return jsonify({
        "result": "Pass" if p >= 0.5 else "Fail",
        "probability": round(p * 100, 1)
    })


# ==========================================
# CONTACT PAGE
# ==========================================

@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":

        new = not os.path.exists(
            os.path.join(BASE, "messages.csv")
        )

        with open(
            os.path.join(BASE, "messages.csv"),
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            w = csv.writer(f)

            if new:
                w.writerow(["Time", "Name", "Email", "Message"])

            w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                request.form["name"],
                request.form["email"],
                request.form["message"]
            ])

        flash("Message sent!")

        return redirect(url_for("contact"))

    return render_template("contact.html")


# ==========================================
# TEACHER LOGIN
# ==========================================

@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():

    if request.method == "POST":

        teacher_id = request.form.get(
            "teacher_id", ""
        ).strip()

        password = request.form.get("password", "")

        conn = get_db()

        teacher = conn.execute("""
            SELECT * FROM teachers
            WHERE teacher_id = ? AND active = 1
        """, (teacher_id,)).fetchone()

        conn.close()

        if teacher and check_password_hash(
            teacher["password_hash"], password
        ):

            session.clear()

            session["teacher_id"] = teacher["teacher_id"]
            session["tname"] = teacher["name"]

            return redirect(url_for("teacher"))

        flash("Invalid teacher ID or password.")

    return render_template("teacher_login.html")



# ==========================================
# TEACHER DASHBOARD
# ==========================================

@app.route("/teacher")
@teacher_required
def teacher():

    df = load_attendance()

    today = datetime.now().strftime("%d-%m-%Y")

    # Get all students
    students = get_students()

    # Get available attendance dates
    dates = sorted(
        df["Date"].dropna().unique().tolist(),
        reverse=True
    )

    if today not in dates:
        dates.insert(0, today)

    # Selected date from dropdown
    selected_date = request.args.get("date", today)

    if selected_date not in dates:
        selected_date = today

    # Attendance records for selected date
    day_df = df[df["Date"] == selected_date]

    total = len(students)

    # Total unique attendance days
    total_days = df["Date"].nunique()

    rows = []

    for roll, name in students.items():

        student_records = df[
            df["Roll No"].astype(str) == str(roll)
        ]

        # Calculate overall attendance percentage
        present_days = student_records[
            student_records["Status"] == "Present"
        ]["Date"].nunique()

        percent = round(
            (present_days / total_days) * 100, 1
        ) if total_days else 0

        # Find attendance on selected date
        record = day_df[
            day_df["Roll No"].astype(str) == str(roll)
        ]

        if not record.empty:

            latest = record.iloc[-1]

            status = latest["Status"]
            time = latest["Time"]

        else:

            status = "Absent"
            time = "-"

        rows.append({
            "roll": roll,
            "name": name,
            "status": status,
            "time": time,
            "percent": percent
        })

    present = sum(
        1 for r in rows if r["status"] == "Present"
    )

    absent = total - present

    return render_template(
        "teacher.html",
        date=selected_date,
        dates=dates,
        rows=rows,
        total=total,
        present=present,
        absent=absent
    )



# ==========================================
# ADMIN: MANAGE TEACHERS
# ==========================================

@app.route("/teacher/manage", methods=["GET", "POST"])
@admin_required
def manage_teachers():

    conn = get_db()

    if request.method == "POST":

        teacher_id = request.form.get(
            "teacher_id", ""
        ).strip()

        name = request.form.get(
            "name", ""
        ).strip()

        password = request.form.get("password", "")

        if not teacher_id or not name or len(password) < 8:

            flash("Enter all fields. Password must have at least 8 characters.")

        elif teacher_id.upper() == "ADMIN":

            flash("ADMIN is reserved for the administrator.")

        else:

            try:

                conn.execute("""
                    INSERT INTO teachers
                    (teacher_id, name, password_hash, active)
                    VALUES (?, ?, ?, 1)
                """, (
                    teacher_id,
                    name,
                    generate_password_hash(password)
                ))

                conn.commit()

                flash("Teacher added successfully.")

            except sqlite3.IntegrityError:

                flash("This Teacher ID already exists.")

    teachers = conn.execute("""
        SELECT teacher_id, name, active
        FROM teachers
        ORDER BY teacher_id
    """).fetchall()

    conn.close()

    return render_template(
        "manage_teachers.html",
        teachers=teachers
    )


# ==========================================
# ADMIN: DEACTIVATE TEACHER
# ==========================================

@app.route("/teacher/deactivate/<teacher_id>", methods=["POST"])
@admin_required
def deactivate_teacher(teacher_id):

    if teacher_id.upper() == "ADMIN":
        flash("Administrator cannot be deactivated.")
        return redirect(url_for("manage_teachers"))

    conn = get_db()

    conn.execute("""
        UPDATE teachers
        SET active = 0
        WHERE teacher_id = ?
    """, (teacher_id,))

    conn.commit()
    conn.close()

    flash("Teacher account deactivated.")

    return redirect(url_for("manage_teachers"))


# ==========================================
# ADMIN: RESET TEACHER PASSWORD
# ==========================================

@app.route("/teacher/reset/<teacher_id>", methods=["POST"])
@admin_required
def reset_teacher_password(teacher_id):

    if teacher_id.upper() == "ADMIN":
        flash("Use your administrator environment settings to reset this account.")
        return redirect(url_for("manage_teachers"))

    password = request.form.get("password", "")

    if len(password) < 8:

        flash("Password must have at least 8 characters.")

    else:

        conn = get_db()

        conn.execute("""
            UPDATE teachers
            SET password_hash = ?, active = 1
            WHERE teacher_id = ?
        """, (
            generate_password_hash(password),
            teacher_id
        ))

        conn.commit()
        conn.close()

        flash("Teacher password reset successfully.")

    return redirect(url_for("manage_teachers"))


# ==========================================
# TEACHER LOGOUT
# ==========================================

@app.route("/teacher/logout")
def teacher_logout():

    session.clear()

    return redirect(url_for("teacher_login"))


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    init_teacher_db()

    create_initial_admin()

    app.run(debug=True)

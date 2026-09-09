# smart_attendance_gui.py
"""
Smart Attendance System - Fixed (Excel backend, blink-to-mark, premium UI)
- Register student (Name + Roll) with face capture
- Blink-to-mark attendance (Mediapipe EAR + face_recognition)
- Attendance saved to attendance.xlsx (sheet "Attendance")
- Daily sheets: Presentees_YYYY-MM-DD (unique), Absentees_YYYY-MM-DD (true absentees)
- Auto-email on Stop (prompts for Gmail App Password)
- Gradient rounded buttons (no lines)
"""

import os
import cv2
import pickle
import threading
import time
import numpy as np
import face_recognition
import mediapipe as mp
import pandas as pd
from datetime import datetime
from tkinter import *
from tkinter import ttk, messagebox, simpledialog, Toplevel
from PIL import Image, ImageTk
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# ---------------- Config ----------------
ATTENDANCE_FILE = "attendance.xlsx"
ENCODINGS_FILE = "encodings.pickle"

# Tweak these if necessary
EAR_THRESHOLD = 0.25
BLINK_CONSEC_FRAMES = 3  # number of frames EAR < threshold to count as blink

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

SENDER_EMAIL = "antoshibin6369@gmail.com"
MENTOR_EMAIL = "antoshibin105402@gmail.com"

# ---------------- File utilities ----------------
def ensure_attendance_file():
    """Create attendance.xlsx with Attendance sheet if missing."""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Roll Number", "Name", "Time", "Date"])
        with pd.ExcelWriter(ATTENDANCE_FILE, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name="Attendance", index=False)

def load_encodings():
    """Return encodings, names, rolls lists (may be empty)."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            encs = data.get("encodings", []) or []
            names = data.get("names", []) or []
            rolls = data.get("rolls", []) or []
            encs = list(encs)
            names = list(names)
            rolls = list(rolls)
            # pad rolls if shorter
            while len(rolls) < len(names):
                rolls.append("")
            return encs, names, rolls
    return [], [], []

def save_encodings(encs, names, rolls):
    """Save encodings, names, rolls to pickle file."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encs, "names": names, "rolls": rolls}, f)

def get_registered_students_df():
    """Return DataFrame with Roll Number and Name for registered students."""
    _, names, rolls = load_encodings()
    df = pd.DataFrame({"Roll Number": [str(r) for r in rolls], "Name": names})
    # keep unique by roll (preserve first)
    df = df.drop_duplicates(subset=["Roll Number"])
    return df

# ---------------- Attendance helpers ----------------
def mark_attendance(roll, name):
    """
    Adds or updates attendance row for (roll, today).
    Uses roll as unique key for the day to prevent duplicates.
    Returns True if newly marked or updated, False if nothing changed.
    """
    ensure_attendance_file()
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    try:
        df_main = pd.read_excel(ATTENDANCE_FILE, sheet_name="Attendance")
    except Exception:
        df_main = pd.read_excel(ATTENDANCE_FILE)

    # ensure columns
    for c in ["Roll Number", "Name", "Time", "Date"]:
        if c not in df_main.columns:
            df_main[c] = ""

    # remove any existing row for this roll & today (prevents duplicates)
    mask_existing = (df_main["Roll Number"].astype(str) == str(roll)) & (df_main["Date"] == today)
    df_main = df_main[~mask_existing]

    new_row = pd.DataFrame([{"Roll Number": str(roll), "Name": name, "Time": now_time, "Date": today}])
    df_main = pd.concat([df_main, new_row], ignore_index=True)

    # write Attendance sheet (replace)
    with pd.ExcelWriter(ATTENDANCE_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_main.to_excel(writer, sheet_name="Attendance", index=False)

    # update today's presentees/absentees
    update_presentees_absentees(df_main, today)
    return True

def update_presentees_absentees(df_main, today):
    """
    Writes two sheets:
     - Presentees_{today}: unique present Roll Number, Name, Time (keeps first occurence)
     - Absentees_{today}: registered students whose roll not in presentees (string compare)
    """
    # ensure Date column
    if "Date" not in df_main.columns:
        df_main["Date"] = ""

    df_today = df_main[df_main["Date"] == today].copy()
    # ensure roll numbers are strings
    if "Roll Number" in df_today.columns:
        df_today["Roll Number"] = df_today["Roll Number"].astype(str)
    else:
        df_today["Roll Number"] = ""

    # drop duplicates by Roll Number (keep first mark)
    presentees = df_today.drop_duplicates(subset=["Roll Number"])[["Roll Number", "Name", "Time"]]

    registered = get_registered_students_df()
    registered["Roll Number"] = registered["Roll Number"].astype(str)

    # compute absentees by string-safe isin (registered rolls not in presentees)
    present_rolls = set(presentees["Roll Number"].astype(str).tolist())
    if len(present_rolls) > 0:
        absentees = registered[~registered["Roll Number"].astype(str).isin(present_rolls)].copy()
    else:
        absentees = registered.copy()

    # ensure consistent columns
    if "Time" not in presentees.columns:
        presentees["Time"] = ""

    # write/replace sheets
    with pd.ExcelWriter(ATTENDANCE_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        if not presentees.empty:
            presentees.to_excel(writer, sheet_name=f"Presentees_{today}", index=False)
        else:
            pd.DataFrame(columns=["Roll Number", "Name", "Time"]).to_excel(writer, sheet_name=f"Presentees_{today}", index=False)

        if not absentees.empty:
            absentees.to_excel(writer, sheet_name=f"Absentees_{today}", index=False)
        else:
            pd.DataFrame(columns=["Roll Number", "Name"]).to_excel(writer, sheet_name=f"Absentees_{today}", index=False)

# ---------------- EAR helper ----------------
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(float(landmarks[i].x * img_w), float(landmarks[i].y * img_h)) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 1.0

# ---------------- Email ----------------
def send_attendance_email(sender_email, sender_password, receiver_email, file_path):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"
        body = "Dear Mentor,\n\nPlease find attached the attendance report (Presentees & Absentees).\n\nRegards,\nSmart Attendance System"
        msg.attach(MIMEText(body, "plain"))

        with open(file_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
        msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print("Email sending failed:", e)
        return False

# ---------------- UI: clean rounded gradient button ----------------
class GradientButton(Canvas):
    """Rounded gradient-like button (single solid color to avoid visible outlines)."""
    def __init__(self, parent, text, command=None, color="#4A90E2", width=230, height=42, radius=20):
        Canvas.__init__(self, parent, width=width, height=height, highlightthickness=0, bg=parent["bg"])
        self.command = command
        self.width = width
        self.height = height
        self.radius = radius
        self.color = color
        self.text = text
        self._draw()
        self.bind("<Button-1>", lambda e: self._on_click())
        # hover effect
        self.bind("<Enter>", lambda e: self._on_enter())
        self.bind("<Leave>", lambda e: self._on_leave())

    def _draw(self, color=None):
        color = color or self.color
        self.delete("all")
        w, h, r = self.width, self.height, self.radius
        # draw rounded rectangle with the same outline color (no thin black lines)
        self.create_arc((0, 0, r * 2, r * 2), start=90, extent=90, fill=color, outline=color)
        self.create_arc((w - r * 2, 0, w, r * 2), start=0, extent=90, fill=color, outline=color)
        self.create_arc((0, h - r * 2, r * 2, h), start=180, extent=90, fill=color, outline=color)
        self.create_arc((w - r * 2, h - r * 2, w, h), start=270, extent=90, fill=color, outline=color)
        self.create_rectangle((r, 0, w - r, h), fill=color, outline=color)
        self.create_rectangle((0, r, w, h - r), fill=color, outline=color)
        self.create_text(w // 2, h // 2, text=self.text, fill="white", font=("Arial", 11, "bold"))

    def _on_click(self):
        if self.command:
            self.command()

    def _on_enter(self):
        # slightly lighten on hover
        self._draw(color=_lighten(self.color, 0.15))

    def _on_leave(self):
        self._draw(color=self.color)

def _lighten(hex_color, amount=0.1):
    """Lighten a hex color by amount (0..1)."""
    hex_color = hex_color.strip("#")
    if len(hex_color) != 6:
        return "#" + hex_color
    r = min(255, int(int(hex_color[0:2], 16) * (1 + amount)))
    g = min(255, int(int(hex_color[2:4], 16) * (1 + amount)))
    b = min(255, int(int(hex_color[4:6], 16) * (1 + amount)))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

# ---------------- Main App ----------------
class SmartAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("1000x650")
        self.root.configure(bg="#E8F0FE")

        ensure_attendance_file()
        self.encs, self.names, self.rolls = load_encodings()
        self.cap = None
        self.running = False
        self.present_today = set()  # holds roll numbers (strings) marked present this session

        # Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.build_ui()

    def build_ui(self):
        left = Frame(self.root, bg="#E8F0FE")
        left.pack(side=LEFT, padx=18, pady=12, fill=Y)

        Label(left, text="SMART ATTENDANCE", font=("Arial", 18, "bold"), bg="#E8F0FE", fg="#16325c").pack(pady=(6, 12))
        Label(left, text="Student Name", bg="#E8F0FE", anchor=W).pack(fill=X)
        self.entry_name = Entry(left, width=28)
        self.entry_name.pack(pady=6)
        Label(left, text="Roll Number", bg="#E8F0FE", anchor=W).pack(fill=X)
        self.entry_roll = Entry(left, width=28)
        self.entry_roll.pack(pady=6)

        # Buttons (colors selected)
        GradientButton(left, "Register Student", command=self.register_student, color="#28A745").pack(pady=8)
        GradientButton(left, "Start Attendance", command=self.start_attendance, color="#4A90E2").pack(pady=8)
        GradientButton(left, "Stop Attendance & Email", command=self.stop_attendance, color="#E74C3C").pack(pady=8)
        GradientButton(left, "Show Records", command=self.show_records, color="#F1C40F").pack(pady=8)
        GradientButton(left, "Exit", command=self.close_app, color="#6C757D").pack(pady=18)

        self.status_var = StringVar(value="Status: Idle")
        Label(left, textvariable=self.status_var, bg="#E8F0FE", anchor=W, fg="#16325c").pack(fill=X, pady=(6,0))

        # Right: camera view
        right = Frame(self.root, bg="#000")
        right.pack(side=RIGHT, padx=12, pady=12, fill=BOTH, expand=True)
        Label(right, text="Camera Feed", bg="#ffffff").pack(anchor=W, padx=8, pady=(6, 2))
        self.video_label = Label(right, bg="#000", width=760, height=520)
        self.video_label.pack(padx=8, pady=8, fill=BOTH, expand=True)

    # ---------------- Register ----------------
    def register_student(self):
        name = self.entry_name.get().strip()
        roll = self.entry_roll.get().strip()
        if not name or not roll:
            messagebox.showwarning("Missing Info", "Please enter Name and Roll Number.")
            return
        if self.running:
            messagebox.showwarning("Stop Attendance", "Stop attendance before registering.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera.")
            return

        messagebox.showinfo("Registration", "Press 'c' to capture face, 'q' to cancel.")
        captured = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Register - Press c to capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                encs_frame = face_recognition.face_encodings(rgb, locs)
                if len(encs_frame) == 0:
                    messagebox.showerror("No Face", "No face detected, try again.")
                else:
                    self.encs.append(encs_frame[0])
                    self.names.append(name)
                    self.rolls.append(str(roll))
                    save_encodings(self.encs, self.names, self.rolls)
                    messagebox.showinfo("Registered", f"{name} ({roll}) registered successfully.")
                    captured = True
                    break
            elif key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured:
            self.entry_name.delete(0, END)
            self.entry_roll.delete(0, END)
            self.status_var.set(f"Registered: {name}")

    # ---------------- Attendance control ----------------
    def start_attendance(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Cannot open camera.")
            return
        # reload encodings to pick up new registrations
        self.encs, self.names, self.rolls = load_encodings()
        # ensure rolls are strings
        self.rolls = [str(r) for r in self.rolls]
        self.present_today.clear()
        self.running = True
        threading.Thread(target=self.video_loop, daemon=True).start()
        self.status_var.set("Status: Running")

    def stop_attendance(self):
        # stop loop
        self.running = False
        time.sleep(0.2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status_var.set("Status: Stopped")
        # prompt for app password to send email
        pwd = simpledialog.askstring("App Password", "Enter Gmail App Password to send report:", show="*")
        if not pwd:
            messagebox.showinfo("Email skipped", "No password entered; email not sent.")
            return
        # update sheets and send email
        try:
            df_main = pd.read_excel(ATTENDANCE_FILE, sheet_name="Attendance")
        except Exception:
            df_main = pd.read_excel(ATTENDANCE_FILE)
        today = datetime.now().strftime("%Y-%m-%d")
        update_presentees_absentees(df_main, today)
        ok = send_attendance_email(SENDER_EMAIL, pwd, MENTOR_EMAIL, ATTENDANCE_FILE)
        if ok:
            messagebox.showinfo("Email Sent", f"Attendance sent to {MENTOR_EMAIL}")
        else:
            messagebox.showerror("Email Failed", "Failed to send email. Check password/network.")

    # ---------------- Records ----------------
    def show_records(self):
        ensure_attendance_file()
        try:
            df = pd.read_excel(ATTENDANCE_FILE, sheet_name="Attendance")
        except Exception:
            df = pd.read_excel(ATTENDANCE_FILE)
        if df.empty:
            messagebox.showinfo("No records", "No attendance recorded yet.")
            return
        # drop duplicates by roll & date for display
        df_display = df.drop_duplicates(subset=["Roll Number", "Date"])
        win = Toplevel(self.root)
        win.title("Attendance Records")
        tree = ttk.Treeview(win, columns=list(df_display.columns), show="headings")
        for c in df_display.columns:
            tree.heading(c, text=c)
            tree.column(c, width=140)
        for _, row in df_display.iterrows():
            tree.insert("", END, values=list(row))
        tree.pack(fill=BOTH, expand=True)

    # ---------------- Video loop ----------------
    def video_loop(self):
        blink_counts = {}    # keyed by roll string
        last_mark_time = {}  # for short confirmation display
        self.status_var.set("Status: Running")

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # detect faces + encodings
            try:
                locs = face_recognition.face_locations(rgb)
                encs_frame = face_recognition.face_encodings(rgb, locs)
            except Exception:
                locs = []
                encs_frame = []

            # mediapipe for eyes
            results = self.mesh.process(rgb)
            meshes = results.multi_face_landmarks if results.multi_face_landmarks else []

            # mesh centroids for matching
            mesh_centroids = []
            for mesh in meshes:
                xs = [lm.x for lm in mesh.landmark]
                ys = [lm.y for lm in mesh.landmark]
                cx = int(np.mean(xs) * w)
                cy = int(np.mean(ys) * h)
                mesh_centroids.append((mesh, cx, cy))

            # iterate face boxes
            for (top, right, bottom, left), enc in zip(locs, encs_frame):
                name, roll = "Unknown", ""
                if len(self.encs) > 0:
                    matches = face_recognition.compare_faces(self.encs, enc, tolerance=0.5)
                    dists = face_recognition.face_distance(self.encs, enc)
                    if len(dists) > 0:
                        best_idx = np.argmin(dists)
                        if matches[best_idx]:
                            name = self.names[best_idx]
                            roll = str(self.rolls[best_idx])

                color = (0, 200, 0) if name != "Unknown" else (10, 90, 200)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{name} ({roll})" if name != "Unknown" else name
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (left, top - 24), (left + tw + 8, top), color, -1)
                cv2.putText(frame, label, (left + 4, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # match mesh
                matched_mesh = None
                for mesh, cx, cy in mesh_centroids:
                    if left <= cx <= right and top <= cy <= bottom:
                        matched_mesh = mesh
                        break

                ear = 1.0
                if matched_mesh is not None:
                    try:
                        left_ear = eye_aspect_ratio(matched_mesh.landmark, LEFT_EYE, w, h)
                        right_ear = eye_aspect_ratio(matched_mesh.landmark, RIGHT_EYE, w, h)
                        ear = (left_ear + right_ear) / 2.0
                        cv2.putText(frame, f"EAR: {ear:.2f}", (left, bottom + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    except Exception:
                        ear = 1.0

                # blink & mark behavior for recognized students only
                if name != "Unknown" and roll != "":
                    cv2.putText(frame, "Blink your eyes to mark attendance", (left, bottom + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    if roll not in blink_counts:
                        blink_counts[roll] = 0
                        last_mark_time[roll] = 0
                    if ear < EAR_THRESHOLD:
                        blink_counts[roll] += 1
                    else:
                        if blink_counts[roll] >= BLINK_CONSEC_FRAMES:
                            # if not already marked this session (prevents duplicates on same run)
                            if roll not in self.present_today:
                                mark_attendance(roll, name)
                                self.present_today.add(roll)
                                last_mark_time[roll] = time.time()
                                self.status_var.set(f"Marked: {name} at {datetime.now().strftime('%H:%M:%S')}")
                        blink_counts[roll] = 0
                    # show "Attendance Marked" for 2s after marking
                    if time.time() - last_mark_time.get(roll, 0) < 2:
                        cv2.putText(frame, "Attendance Marked", (left, top - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Unknown - register to mark", (left, bottom + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            # show hint if no faces detected
            if len(locs) == 0:
                cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

            # display frame in Tkinter label (converted to PhotoImage)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # optionally resize to fit the label (keeps aspect)
            img_w, img_h = img.size
            max_w, max_h = 760, 520
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            if scale < 1.0:
                img = img.resize((int(img_w * scale), int(img_h * scale)))
            imgtk = ImageTk.PhotoImage(img)
            self.video_label.imgtk = imgtk  # keep ref
            self.video_label.configure(image=imgtk)

        # cleanup
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status_var.set("Status: Stopped")

    # ---------------- Close app ----------------
    def close_app(self):
        self.running = False
        time.sleep(0.15)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()

# ---------------- Run ----------------
if __name__ == "__main__":
    root = Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()
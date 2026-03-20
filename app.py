from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import mysql.connector
from datetime import datetime
from paddleocr import PaddleOCR
import json
from werkzeug.security import generate_password_hash, check_password_hash

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

monitoring = False
cap = None
model = None
ocr = None
startTime = None
license_plates = set()
count = 0
location = "Main Entrance"  
camera_on_time = None
camera_history = []

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

def init_models():
    global model, ocr
    model = YOLO("weights/best.pt")
    ocr = PaddleOCR(use_textline_orientation=True)

def paddle_ocr(frame, x1, y1, x2, y2):
    if y2 <= y1 or x2 <= x1:
        return ""
    frame_crop = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame_crop)
    if not result or result[0] is None:
        return ""
    text = ""
    for line in result:
        if line is None:
            continue
        for box, (t, s) in line:
            if s > 0.6:
                text = t
    pattern = re.compile(r'[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)

def save_detection(plate, date, time, location, start_time, end_time, image):
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections(license_plate, date, time, location, monitoring_start, monitoring_end, image)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    ''', (plate, date, time, location, start_time, end_time, image))
    conn.commit()
    conn.close()

def log_camera_action(action):
    global camera_on_time, camera_history
    now = datetime.now()
    if action == "on":
        camera_on_time = now
        camera_history.append({"action": "on", "timestamp": now, "duration": 0})
    elif action == "off" and camera_on_time:
        duration = (now - camera_on_time).seconds
        camera_history[-1]["duration"] = duration
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO camera_history(action, timestamp, duration)
            VALUES (%s, %s, %s)
        ''', ("on", camera_on_time, duration))
        conn.commit()
        conn.close()

def gen_frames():
    global monitoring, cap, startTime, license_plates, count
    while monitoring:
        ret, frame = cap.read()
        if not ret:
            break
        currentTime = datetime.now()
        count += 1
        results = model.predict(frame, conf=0.15)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)
                    # Trigger notification
                    ret_img, buffer = cv2.imencode('.jpg', frame)
                    image = buffer.tobytes()
                    save_detection(label, currentTime.date(), currentTime.time(), location, startTime, currentTime, image)
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            # Save to JSON as before
            interval_data = {
                "Start Time": startTime.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "End Time": endTime.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                "License Plate": list(license_plates)
            }
            cummulative_file_path = "json/LicensePlateData.json"
            if os.path.exists(cummulative_file_path):
                with open(cummulative_file_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            existing_data.append(interval_data)
            with open(cummulative_file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            startTime = currentTime
            license_plates.clear()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@login_required
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Validate username: First letter capital, only letters
        if not re.match(r'^[A-Z][a-zA-Z]*$', username):
            flash('Username must start with a capital letter and contain only letters')
            return render_template('login.html')
        
        # Validate password: Exactly 6 characters
        if not re.match(r'^.{6}$', password):
            flash('Password must be exactly 6 characters')
            return render_template('login.html')
        
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1])
            login_user(user_obj)
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Validate username: First letter capital, only letters
        if not re.match(r'^[A-Z][a-zA-Z]*$', username):
            flash('Username must start with a capital letter and contain only letters')
            return render_template('signup.html')
        
        # Validate password: Exactly 6 characters
        if not re.match(r'^.{6}$', password):
            flash('Password must be exactly 6 characters')
            return render_template('signup.html')
        
        hashed_password = generate_password_hash(password)
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            flash('Account created successfully')
            return redirect(url_for('login'))
        except:
            flash('Username already exists')
        conn.close()
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/camera_capture')
@login_required
def camera_capture():
    return render_template('camera_capture.html')

@app.route('/live_monitoring')
@login_required
def live_monitoring():
    return render_template('live_monitoring.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    # Validate new password format
    if not re.match(r'^.{6}$', new_password):
        flash('Password must be exactly 6 characters')
        return redirect(url_for('profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match')
        return redirect(url_for('profile'))
    
    # Verify current password
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE id = %s", (current_user.id,))
    user = cursor.fetchone()
    
    if not user or not check_password_hash(user[0], current_password):
        flash('Current password is incorrect')
        conn.close()
        return redirect(url_for('profile'))
    
    # Update password
    hashed_password = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password, current_user.id))
    conn.commit()
    conn.close()
    
    flash('Password updated successfully')
    return redirect(url_for('profile'))

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users WHERE id = %s", (current_user.id,))
        conn.commit()
        conn.close()
        logout_user()
        flash('Account deleted successfully')
        return redirect(url_for('login'))
    except Exception as e:
        conn.close()
        flash('Error deleting account')
        return redirect(url_for('profile'))

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if request.method == 'POST':
        pin = request.form['pin']
        if pin == 'admin123':
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Invalid PIN')
    return render_template('admin.html')

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('admin'))
    return render_template('admin_dashboard.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_monitoring', methods=['POST'])
@login_required
def start_monitoring():
    global monitoring, cap, startTime, license_plates, count, location
    # camera index may be provided in JSON body
    camera_idx = 0
    try:
        data = request.get_json(silent=True)
        if data and 'camera' in data:
            camera_idx = int(data['camera'])
    except Exception:
        camera_idx = 0

    # Set location based on camera selection
    camera_locations = {
        0: "Camera 1 - Main Entrance",
        1: "Camera 2 - Side Entrance"
    }
    location = camera_locations.get(camera_idx, "Main Entrance")

    # if already monitoring with a different camera, release existing
    if monitoring and cap:
        try:
            cap.release()
        except Exception:
            pass

    monitoring = True
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        monitoring = False
        cap.release()
        return jsonify({'status': 'error', 'message': f'Cannot open camera {camera_idx}'}), 500

    startTime = datetime.now()
    license_plates = set()
    count = 0
    init_models()
    log_camera_action("on")
    return jsonify({'status': 'started'})

@app.route('/stop_monitoring', methods=['POST'])
@login_required
def stop_monitoring():
    global monitoring, cap
    monitoring = False
    if cap:
        cap.release()
        log_camera_action("off")
    return jsonify({'status': 'stopped'})

@app.route('/get_live_data')
@login_required
def get_live_data():
    # Return current detections
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    cursor.execute("SELECT license_plate, date, time, location, monitoring_start, monitoring_end FROM detections ORDER BY id DESC LIMIT 10")
    data = cursor.fetchall()
    conn.close()
    return jsonify([{
        'plate': row[0],
        'date': str(row[1]),
        'time': str(row[2]),
        'location': row[3],
        'start': str(row[4]),
        'end': str(row[5])
    } for row in data])

@app.route('/search_plate')
@login_required
def search_plate():
    plate = request.args.get('plate', '').strip()
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    if plate:
        cursor.execute(
            "SELECT license_plate, date, time, location, monitoring_start, monitoring_end FROM detections WHERE license_plate LIKE %s ORDER BY id DESC",
            (f"%{plate}%",)
        )
    else:
        cursor.execute(
            "SELECT license_plate, date, time, location, monitoring_start, monitoring_end FROM detections ORDER BY id DESC LIMIT 100"
        )
    data = cursor.fetchall()
    conn.close()
    return jsonify([{
        'plate': row[0],
        'date': str(row[1]),
        'time': str(row[2]),
        'location': row[3],
        'start': str(row[4]),
        'end': str(row[5])
    } for row in data])

@app.route('/download_plate_history')
@login_required
def download_plate_history():
    plate = request.args.get('plate', '').strip()
    conn = mysql.connector.connect(host="localhost", user="root", password="", database="projectx")
    cursor = conn.cursor()
    if plate:
        cursor.execute(
            "SELECT license_plate, date, time, location, monitoring_start, monitoring_end FROM detections WHERE license_plate LIKE %s ORDER BY id DESC",
            (f"%{plate}%",)
        )
    else:
        cursor.execute(
            "SELECT license_plate, date, time, location, monitoring_start, monitoring_end FROM detections ORDER BY id DESC"
        )
    rows = cursor.fetchall()
    conn.close()

    from io import StringIO
    import csv

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['License Plate', 'Date', 'Time', 'Location', 'Monitoring Start', 'Monitoring End'])
    for row in rows:
        writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])

    response = Response(buffer.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = f'attachment; filename="plate_history_{plate or "all"}.csv"'
    return response

@app.route('/get_historical_data')
@login_required
def get_historical_data():
    try:
        with open('json/LicensePlateData.json', 'r') as f:
            data = json.load(f)
        data.sort(key=lambda x: x['Start Time'], reverse=True)
        return jsonify(data)
    except:
        return jsonify([])

@app.route('/capture_single')
@login_required
def capture_single():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, conf=0.15)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    now = datetime.now()
                    ret_img, buffer = cv2.imencode('.jpg', frame)
                    image = buffer.tobytes()
                    save_detection(label, now.date(), now.time(), location, now, now, image)
                    cap.release()
                    return jsonify({'plate': label, 'date': str(now.date()), 'time': str(now.time()), 'location': location})
    cap.release()
    return jsonify({'error': 'No plate detected'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

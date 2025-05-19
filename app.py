from flask import Flask, render_template, request, redirect, url_for, session, g, flash
import sqlite3
import re
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests

# App configuration
app = Flask(__name__)
app.secret_key = 'Joy@#12345'

# Database and upload folder settings
DATABASE = 'geeklogin.db'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Activity label mapping for predictions
label_map = {
    "1": "walking",
    "2": "running",
    "3": "sitting",
    "4": "standing",
    "5": "upstairs",
    "6": "downstairs"
}

# Database connection helpers

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext

def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize database with accounts table

def init_db():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT NOT NULL
            );
            '''
        )
        conn.commit()
        conn.close()

init_db()

# Routes

@app.route('/', methods=['GET'])
def home():
    if session.get('loggedin'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('loggedin'):
        return redirect(url_for('dashboard'))

    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT * FROM accounts WHERE username = ?', (username,))
        account = cursor.fetchone()

        if account and check_password_hash(account['password'], password):
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            msg = 'Incorrect username or password!'
            flash(msg, 'danger')

    return render_template('login.html', msg=msg)


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        db = get_db()
        cursor = db.cursor()

        # Validation checks
        cursor.execute('SELECT 1 FROM accounts WHERE username = ?', (username,))
        if cursor.fetchone():
            msg = 'Account already exists!'
            flash(msg, 'warning')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            flash(msg, 'warning')
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only letters and numbers!'
            flash(msg, 'warning')
        elif not (username and password and email):
            msg = 'Please fill out all fields!'
            flash(msg, 'warning')
        else:
            hashed_password = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO accounts (username, password, email) VALUES (?, ?, ?)',
                (username, hashed_password, email)
            )
            db.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html', msg=msg)


@app.route('/dashboard')
def dashboard():
    if not session.get('loggedin'):
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('loggedin'):
        return redirect(url_for('login'))

    summary = None
    if request.method == 'POST':
        acc_file = request.files.get('acc_file')
        gyro_file = request.files.get('gyro_file')

        if not acc_file or not gyro_file:
            flash('Please upload both Accelerometer and Gyroscope CSV files!', 'warning')
        else:
            # Save uploads
            acc_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(acc_file.filename))
            gyro_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(gyro_file.filename))
            acc_file.save(acc_path)
            gyro_file.save(gyro_path)

            try:
                resp = requests.post(
                    'http://localhost:8000/predict/',
                    files={'acc_file': open(acc_path, 'rb'), 'gyro_file': open(gyro_path, 'rb')}
                )
                resp.raise_for_status()
                data = resp.json().get('activity_summary_seconds', {})
                summary = {label_map.get(str(k), str(k)): v for k, v in data.items()}
            except Exception as e:
                flash(f'Prediction error: {e}', 'danger')

    return render_template('predict.html', summary=summary)


if __name__ == '__main__':
    app.run(debug=True)

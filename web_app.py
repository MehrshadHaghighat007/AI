from flask import Flask, request, render_template, session, redirect, url_for, flash, send_from_directory, make_response, jsonify
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file (not committed to git)
# Must run BEFORE importing rag_assistant, since that module reads
# RAW_INDEX_FILE / RAW_REFS_FILE / LLM_ENDPOINT / LLM_MODEL_NAME from
# os.environ at import time (module-level code). Importing it earlier
# would silently fall back to this module's hardcoded defaults instead
# of the values in .env.
from rag_assistant import RAGAssistant
import unicodedata  # Added for safe_filename

app = Flask(__name__)
app.secret_key = os.environ["SECRET_KEY"]
app.config['UPLOAD_FOLDER'] = os.environ["UPLOAD_FOLDER"]
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
# === ADD THIS AFTER app.config ===
USER_CHAT_DIR = os.environ["USER_CHAT_DIR"]
os.makedirs(USER_CHAT_DIR, exist_ok=True)  # Safe new folder

# Load employees.json
EMPLOYEES_FILE = os.environ["EMPLOYEES_FILE"]
try:
    with open(EMPLOYEES_FILE, "r") as f:
        employees = json.load(f)
except FileNotFoundError:
    print(f"Error: {EMPLOYEES_FILE} not found")
    employees = {}

assistant = RAGAssistant(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def safe_filename(filename):
    """Unicode-safe filename sanitizer that preserves Persian/Unicode characters."""
    filename = unicodedata.normalize('NFKC', filename)
    # Remove forbidden filesystem characters
    forbidden_chars = r'\/:*?"<>|'
    filename = ''.join(c for c in filename if c not in forbidden_chars)
    # Optional: truncate if too long (e.g., 255 chars max for some FS)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    return filename.strip()

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        if employees.get(session['user_id'], {}).get('role') == 'head_of_unit':
            return redirect(url_for('select_role'))
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user_id = request.form['user_id'].strip()
        if user_id in employees:
            session['user_id'] = user_id
            session['user_data'] = employees[user_id]
            if employees[user_id].get('role') == 'head_of_unit':
                return redirect(url_for('select_role'))
            return redirect(url_for('dashboard'))
        flash('Invalid ID', 'danger')
    return render_template('login.html')

@app.route('/select-role', methods=['GET', 'POST'])
def select_role():
    if 'user_id' not in session or employees.get(session['user_id'], {}).get('role') != 'head_of_unit':
        return redirect(url_for('login'))
    if request.method == 'POST':
        role_choice = request.form['role_choice']
        if role_choice == 'head':
            return redirect(url_for('head_dashboard'))
        return redirect(url_for('employee_dashboard'))
    return render_template('select_role.html')

@app.route('/switch-role')
def switch_role():
    if 'user_id' not in session or employees[session['user_id']].get('role') != 'head_of_unit':
        return redirect(url_for('login'))
    return redirect(url_for('select_role'))

@app.route('/logout')
def logout():
    session.clear()  # Only session
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_data = employees[session['user_id']]
    if user_data.get('role') == 'head_of_unit':
        return redirect(url_for('select_role'))
    return redirect(url_for('employee_dashboard'))

@app.route('/head', methods=['GET', 'POST'])
def head_dashboard():
    if 'user_id' not in session or employees[session['user_id']].get('role') != 'head_of_unit':
        return redirect(url_for('login'))
    log = []
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]  # Filter to PDFs only
    doc_count = len(files)
    log_content = ""
    feedback_content = ""

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'upload':
            files_upload = request.files.getlist('files')
            for file in files_upload:
                if file and allowed_file(file.filename):
                    filename = safe_filename(file.filename)  # Changed to safe_filename
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.exists(full_path):
                        log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User {session['user_id']} overwritten: {filename}")
                    else:
                        log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User {session['user_id']} added: {filename}")
                    file.save(full_path)
            message = 'Files uploaded successfully! Restart app to update indexes.'
            if is_ajax:
                files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
                return jsonify({'success': True, 'message': message, 'files': files})
            else:
                flash(message, 'success')
        elif action == 'remove':
            remove_files = request.form.getlist('remove_files')
            for file_name in remove_files:
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User {session['user_id']} removed: {file_name}")
            message = 'Files removed successfully! Restart app to update indexes.'
            if is_ajax:
                files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.lower().endswith('.pdf')]
                return jsonify({'success': True, 'message': message, 'files': files})
            else:
                flash(message, 'success')
        elif action == 'view_log':
            log_file = os.path.join(app.config['UPLOAD_FOLDER'], 'upload_log.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
            else:
                log_content = "No log file found."
        elif action == 'view_feedback':
            feedback_file = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback_log.txt')
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_content = f.read()
            else:
                feedback_content = "No feedback log file found."

        if log:
            log_file = os.path.join(app.config['UPLOAD_FOLDER'], 'upload_log.txt')
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                for entry in log:
                    f.write(entry + '\n')
                f.write(f"{'-' * 50}\n")

    return render_template('head_dashboard.html', files=files, log_content=log_content, feedback_content=feedback_content, doc_count=doc_count)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        feedback_text = request.form.get('feedback', '').strip()
        if feedback_text:
            feedback_log = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback_log.txt')
            with open(feedback_log, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User {session['user_id']}: {feedback_text}\n")
            flash('Feedback submitted successfully', 'success')
            return redirect(url_for('employee_dashboard'))  # Redirect back to dashboard

    # GET: Show form
    return render_template('feedback.html')

@app.route('/employee', methods=['GET', 'POST'])
def employee_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user_data = session['user_data']

    # Load conversations from DB
    conversations = assistant.get_conversations(user_id)

    # Handle current conversation
    current_conv_id = session.get('current_conversation_id')
    action = request.args.get('action')
    conv_id_param = request.args.get('conv_id')

    if action == 'new_chat':
        current_conv_id = None
        session['current_conversation_id'] = None
    elif conv_id_param:
        current_conv_id = int(conv_id_param)
        session['current_conversation_id'] = current_conv_id

    if current_conv_id:
        current_chat = assistant.get_conversation_searches(current_conv_id)
    else:
        current_chat = []

    # Handle POST (query submission) for non-AJAX fallback
    if request.method == 'POST' and not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        query = request.form.get('query', '').strip()
        if not query:
            flash('No query provided.', 'danger')
            return redirect(url_for('employee_dashboard'))

        was_new = current_conv_id is None
        reply, conv_id = assistant.process_query(
            query, user_id,
            user_name=user_data['name'],
            user_unit=user_data['unit'],
            conversation_id=current_conv_id
        )

        if was_new:
            session['current_conversation_id'] = conv_id

        return redirect(url_for('employee_dashboard'))

    # Handle AJAX
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        data = request.get_json()  # CHANGED TO request.get_json()
        query = data.get('query', '').strip() if data else ''
        if not query:
            return jsonify({'error': 'No query provided.'}), 400

        try:
            current_conv_id = session.get('current_conversation_id')
            was_new = current_conv_id is None
            reply, conv_id = assistant.process_query(
                query, user_id,
                user_name=user_data['name'],
                user_unit=user_data['unit'],
                conversation_id=current_conv_id
            )

            if was_new:
                session['current_conversation_id'] = conv_id
                title = query[:50]
                time_str = datetime.now().strftime('%H:%M')
            else:
                title = None
                time_str = None

            ai_html = f"""
            <div class="d-flex justify-content-start mb-4">
                <div class="bg-white p-3 rounded-3 shadow-sm border" style="max-width: 80%;">
                    <div class="d-flex align-items-center mb-2">
                        <div class="bg-primary text-white rounded-circle d-flex align-items-center justify-content-center me-2" style="width: 38px; height: 38px;">
                            AI
                        </div>
                        <strong class="text-primary">AI Assistant</strong>
                    </div>
                    <div class="response-text">{reply}</div>
                </div>
            </div>
            """

            return jsonify({
                'ai_html': ai_html,
                'new_conv': was_new,
                'title': title,
                'time': time_str,
                'conv_id': conv_id if was_new else None
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    common_searches = assistant.get_common_searches(user_id, top_n=10) or []

    return render_template(
        'employee_dashboard.html',
        user_data=user_data,
        current_chat=current_chat,
        conversations=conversations,
        common_searches=common_searches
    )

@app.route('/new-chat')
def new_chat():
    return redirect(url_for('employee_dashboard') + '?action=new_chat')


@app.route('/pdfs/<filename>')
def serve_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download-log')
def download_log():
    if 'user_id' not in session or employees[session['user_id']].get('role') != 'head_of_unit':
        return redirect(url_for('login'))

    log_file = os.path.join(app.config['UPLOAD_FOLDER'], 'upload_log.txt')
    doc_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')])

    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    else:
        log_content = "No log file found."

    log_content = f"Document Count: {doc_count}\n\n{log_content}"

    response = make_response(log_content)
    response.headers['Content-Disposition'] = 'attachment; filename=operation_log.txt'
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    return response

@app.route('/download-feedback')  # New route
def download_feedback():
    if 'user_id' not in session or employees[session['user_id']].get('role') != 'head_of_unit':
        return redirect(url_for('login'))

    feedback_file = os.path.join(app.config['UPLOAD_FOLDER'], 'feedback_log.txt')

    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedback_content = f.read()
    else:
        feedback_content = "No feedback log file found."

    response = make_response(feedback_content)
    response.headers['Content-Disposition'] = 'attachment; filename=feedback_log.txt'
    response.headers['Content-Type'] = 'text/plain; charset=utf-8'
    return response

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)

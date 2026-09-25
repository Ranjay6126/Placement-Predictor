# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
import re
from pathlib import Path
from datetime import datetime
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Initialize Flask app
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'Frontend'
app = Flask(__name__, template_folder=str(FRONTEND_DIR / 'templates'), static_folder=str(FRONTEND_DIR / 'static'))
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'placement_model.pkl'
ENCODERS_PATH = MODEL_DIR / 'encoders.pkl'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'placement-predictor-development-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_PERMANENT'] = True

# Gemini is optional. Set GEMINI_API_KEY in the environment to enable it.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# Initialize database with SQLAlchemy
db = SQLAlchemy(app)

# Initialize login manager for handling user sessions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load user from database by ID
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Signup route (GET for form display, POST for form submission)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if username or email already exists
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()

        if user_exists:
            flash('Username already exists!')
            return redirect(url_for('signup'))

        if email_exists:
            flash('Email already exists!')
            return redirect(url_for('signup'))

        # Create a new user with hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully!')
        return redirect(url_for('login'))

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Find user by username
        user = User.query.filter_by(username=username).first()

        # Check credentials (support legacy sha256 hashes too)
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        # Log the user in
        login_user(user)
        return redirect(url_for('predict'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Prediction route (requires login)
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            # Load trained model and encoders
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)

            # Get form inputs
            sl_no = int(request.form.get('sl_no'))
            gender = request.form.get('gender')
            ssc_p = float(request.form.get('ssc_p'))
            ssc_b = request.form.get('ssc_b')
            hsc_p = float(request.form.get('hsc_p'))
            hsc_b = request.form.get('hsc_b')
            hsc_s = request.form.get('hsc_s')
            degree_p = float(request.form.get('degree_p'))
            degree_t = request.form.get('degree_t')
            workex = request.form.get('workex')
            etest_p = float(request.form.get('etest_p'))
            specialisation = request.form.get('specialisation')
            mba_p = float(request.form.get('mba_p'))

            # Prepare DataFrame for prediction
            input_data = pd.DataFrame({
                'sl_no': [sl_no],
                'gender': [gender],
                'ssc_p': [ssc_p],
                'ssc_b': [ssc_b],
                'hsc_p': [hsc_p],
                'hsc_b': [hsc_b],
                'hsc_s': [hsc_s],
                'degree_p': [degree_p],
                'degree_t': [degree_t],
                'workex': [workex],
                'etest_p': [etest_p],
                'specialisation': [specialisation],
                'mba_p': [mba_p]
            })

            # Encode categorical values
            categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
            for col in categorical_cols:
                if col in encoders:
                    input_data[col] = encoders[col].transform(input_data[col])

            # Perform prediction and retain the information needed by recommendations.
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = probabilities[1] if prediction == 'Placed' else probabilities[0]
            session['prediction_data'] = {
                'prediction': prediction,
                'confidence': round(float(confidence) * 100, 2),
                'input_data': request.form.to_dict()
            }

            return redirect(url_for('show_result'))

        except Exception as e:
            flash(f'Error in prediction: {str(e)}')
            return redirect(url_for('predict'))

    return render_template('predict.html')

# Dedicated route to display the saved prediction result from session
@app.route('/result')
@login_required
def show_result():
    prediction_data = session.get('prediction_data')
    if not prediction_data:
        flash('Please make a prediction first.')
        return redirect(url_for('predict'))
    return render_template(
        'result.html',
        prediction=prediction_data['prediction'],
        confidence=prediction_data['confidence']
    )

@app.route('/recommendation')
@login_required
def recommendation():
    prediction_data = session.get('prediction_data')
    if not prediction_data:
        flash('Please make a prediction first.')
        return redirect(url_for('predict'))

    model = joblib.load(MODEL_PATH)
    feature_names = ['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']
    labels = {
        'ssc_p': 'Secondary Education %', 'hsc_p': 'Higher Secondary %',
        'degree_p': 'Degree Percentage', 'mba_p': 'MBA Percentage',
        'etest_p': 'Employability Test %', 'workex': 'Work Experience',
        'specialisation': 'MBA Specialization'
    }
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True)[:5]
    feature_importances = [{'feature': labels.get(name, name.replace('_', ' ').title()), 'importance': round(float(value) * 100, 1)} for name, value in ranked]
    return render_template('recommendation.html', prediction=prediction_data['prediction'], confidence=prediction_data['confidence'], feature_importances=feature_importances, input_data=prediction_data['input_data'])

# ---------- Smart Placement Chatbot ----------

PLACEMENT_KEYWORDS = {
    'resume': ['resume', 'cv', 'curriculum vitae', 'biodata', 'resume format', 'resume template', 'cover letter'],
    'interview': ['interview', 'hr round', 'technical round', 'gd', 'group discussion', 'aptitude test',
                  'interview prep', 'interview question', 'technical interview', 'hr interview',
                  'telephonic interview', 'virtual interview', 'panel interview'],
    'placement': ['placement', 'placed', 'placement drive', 'campus drive', 'job', 'hiring', 'recruit',
                  'recruitment', 'on campus', 'off campus', 'pool campus', 'tier 1', 'tier company'],
    'career': ['career', 'career path', 'career guide', 'career option', 'future', 'job role',
               'job profile', 'domain', 'field', 'specialization', 'stream'],
    'skills': ['skill', 'technical skill', 'soft skill', 'communication', 'coding', 'programming',
               'python', 'java', 'dsa', 'data structure', 'algorithm', 'dbms', 'sql', 'web dev',
               'development', 'project', 'portfolio', 'certification', 'course'],
    'academics': ['academic', 'percentage', 'cgpa', 'marks', 'grade', 'ssc', 'hsc', '10th', '12th',
                  'degree', 'btech', 'bcom', 'bsc', 'mba', 'education', 'backlog', 'gap year'],
    'company': ['company', 'ctc', 'salary', 'package', 'stipend', 'internship', 'intern', 'ppo',
                'pre placement offer', 'amazon', 'google', 'microsoft', 'tcs', 'infosys', 'wipro',
                'accenture', 'cognizant', 'capgemini', 'ibm', 'deloitte', 'pwc', 'ey', 'kpmg'],
    'preparation': ['prepare', 'preparation', 'study plan', 'roadmap', 'strategy', 'tips', 'trick',
                    'guidance', 'how to', 'how can i', 'suggest', 'recommend', 'improve', 'boost']
}

OFF_TOPIC_RESPONSE = (
    "I'm a Placement Assistant specialized only in placement guidance, "
    "resume building, interview preparation, career planning, and campus recruitment topics.\n\n"
    "Feel free to ask me about:\n"
    "• Resume / CV writing tips\n"
    "• Interview preparation (HR, Technical, GD rounds)\n"
    "• Aptitude and coding preparation strategy\n"
    "• Career paths and domain selection\n"
    "• Skills, certifications and projects\n"
    "• Placement drives, CTC, internships and PPOs\n"
    "• Academic performance and improvement tips"
)

def is_placement_related(message):
    """Strict keyword-based filter to allow only placement-related questions."""
    text = message.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    for category, keywords in PLACEMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return True, category
    return False, None

def smart_local_chat_response(message, category):
    """Category-aware local fallback responses for placement queries."""
    text = message.lower()

    if category == 'resume':
        return (
            "Resume building tips:\n"
            "1. Keep it 1 page (2 pages max for >5 yr experience)\n"
            "2. Lead with Skills → Projects → Experience → Education\n"
            "3. Use action verbs (Built, Designed, Implemented, Optimized)\n"
            "4. Quantify every achievement (\"Improved latency by 30%\" not \"Worked on performance\")\n"
            "5. Tailor keywords to the JD (job description)\n"
            "6. Keep a strong Summary / Objective line (2–3 lines only)\n"
            "7. No photo, no gender/religion info unless the company asks\n"
            "8. Export as PDF with filename: YourName_Resume.pdf"
        )
    if category == 'interview':
        if any(w in text for w in ['gd', 'group']):
            return (
                "Group Discussion (GD) tips:\n"
                "1. Initiate the discussion if possible — it gives extra points\n"
                "2. Speak clearly, use facts/examples instead of opinion\n"
                "3. Listen, give others a chance, and politely counter\n"
                "4. Keep a balanced view (don't take an extreme stand)\n"
                "5. Summarize the group points at the end if no one else does\n"
                "6. Avoid slang, interrupting, or dominating the talk"
            )
        if any(w in text for w in ['hr', 'behavior']):
            return (
                "HR Interview tips:\n"
                "1. Prepare a 60-second intro: Name → College → Degree → Skills → 1 project highlight → Goal\n"
                "2. Use STAR method for behavioral questions (Situation → Task → Action → Result)\n"
                "3. Prepare answers for: strengths/weaknesses, why this company, where do you see yourself in 5 years\n"
                "4. Be honest about gaps/backlogs — frame them as learning experiences\n"
                "5. Have 1–2 thoughtful questions ready to ask the interviewer\n"
                "6. Dress formally, reach 10 min early, and maintain eye contact + smile"
            )
        if any(w in text for w in ['technical', 'coding', 'dsa']):
            return (
                "Technical Interview roadmap:\n"
                "1. DSA → Arrays, Strings, LL, Stacks/Queues, Trees, Graphs, DP (LeetCode 150 / GFG SDE Sheet)\n"
                "2. Core subjects: OOP, DBMS (SQL joins/indexes/ACID), OS, CN\n"
                "3. Language-specific concepts (Java — collections/multithreading / Python — decorators/generators)\n"
                "4. Project deep-dive: architecture, tech stack choices, bottlenecks, what you'd improve\n"
                "5. System design basics: LLD (classes/interfaces) + basics of HLD for seniors\n"
                "6. Practice writing clean code on paper / whiteboard + talk-aloud while solving"
            )
        return (
            "Interview preparation plan:\n"
            "• Day 1–5: Revise your DSA basics & language fundamentals\n"
            "• Day 6–10: Core subjects (OOP, DBMS, OS, CN)\n"
            "• Day 11–14: Prepare 2 project explanations end-to-end\n"
            "• Day 15+: Daily 1 mock interview + 2 medium coding + HR questions\n"
            "Maintain an interview notebook for each company — note patterns, rounds asked, mistakes made."
        )
    if category == 'company':
        if 'internship' in text or 'intern' in text:
            return (
                "Internship strategy:\n"
                "1. Apply on Internshala, LinkedIn, AngelList, company career pages\n"
                "2. Off-campus: cold mail HRs with a short, specific pitch + resume + 1 project link\n"
                "3. Your LinkedIn photo/bio/headline must be professional\n"
                "4. Aim for 20–30 applications/day, not 1 mass blast\n"
                "5. 3–6 month internships look great on resume and can convert to PPO\n"
                "6. Contribute to open source or build personal mini-projects if no internship yet"
            )
        if any(w in text for w in ['ctc', 'salary', 'package']):
            return (
                "CTC / Salary negotiation:\n"
                "• CTC = Base + HRA + Allowances + PF + Bonus + Stocks (NOT equal to in-hand)\n"
                "• In-hand in India is roughly 65–80% of CTC depending on structure\n"
                "• Tier-1 companies: 8–45 LPA CTC (fresher)\n"
                "• Mass recruiters (TCS/Infosys/Wipro/Cognizant): 3.2–7 LPA CTC\n"
                "• Service-based MNCs (Accenture/Capgemini/IBM): 4–10 LPA CTC\n"
                "• Product-based / FAANG tier: 12–45+ LPA CTC\n"
                "During negotiation — quote based on market, not your needs; always have another offer as backup."
            )
        return (
            "For company-specific preparation:\n"
            "1. Visit Glassdoor / GFG Interview Experiences — read last 10 reviews for that company\n"
            "2. Check the company's website for their products, tech stack, values, recent news\n"
            "3. Pattern varies: Mass recruiters → Aptitude + Essay + 2–3 rounds\n"
            "   Product-based → 2 coding + 2–3 technical + 1 HR\n"
            "4. Prepare \"Why do you want to join us?\" with specific reasons from their website/news\n"
            "Tell me which company's drive you're targeting and I'll tailor the tips!"
        )
    if category == 'skills' or 'dsa' in text or 'coding' in text:
        if 'project' in text:
            return (
                "Projects to boost placement resume:\n"
                "For CS/IT:\n"
                "• 1 Full-stack (MERN / Django + React) — deploy on Vercel/Railway\n"
                "• 1 Data Science / ML project with a clean dashboard + public GitHub\n"
                "• 1 System-design mini (URL shortener, Todo API with auth + rate limit)\n"
                "For Non-CS:\n"
                "• 1 Excel / BI dashboard (Tableau/PowerBI)\n"
                "• 1 case-study report on a recent business problem\n"
                "Rule: Quality > Quantity. Have 3 polished projects instead of 10 half-done ones."
            )
        return (
            "High-demand skills (2025) by category:\n"
            "• Programming: Python / Java / JS (any 2 deeply)\n"
            "• DSA: LeetCode 150 → top 200 company tagged\n"
            "• Dev basics: Git + GitHub, Linux CLI, REST API basics, SQL joins\n"
            "• Frameworks: React + Node (web), Spring Boot (Java), Django/Flask (Py)\n"
            "• Databases: MySQL / PostgreSQL + Redis basics\n"
            "• Soft: Written + verbal communication, quick mental math for aptitude\n\n"
            "Pick 1 role (Backend/Frontend/Fullstack/Data/Analyst/QA/DevOps) — don't be a jack of all!"
        )
    if category == 'academics':
        return (
            "Academic & eligibility tips:\n"
            "• Most companies have 60% / 6.5 CGPA cutoff across 10th, 12th, and Degree\n"
            "• 1–2 year gaps are usually OK if you justify them (upskilling, project, internship)\n"
            "• 1 active backlog often blocks eligibility — clear it before drives start\n"
            "• If your percentage is low:\n"
            "   1) Focus heavily on projects + coding skills to compensate\n"
            "   2) Apply off-campus to companies that don't have strict cutoffs\n"
            "   3) Get certifications and internships on the resume\n"
            "• MBA students: CAT/CMAT score + summer internships + case competitions matter a lot"
        )
    if category == 'preparation':
        return (
            "6-Month Placement Preparation Roadmap:\n"
            "Month 1: Math & English aptitude basics + 1 programming language fundamentals\n"
            "Month 2: DSA — Arrays, Strings, Searching, Sorting, HashMaps\n"
            "Month 3: DSA — LinkedList, Stacks, Queues, Trees, Recursion + build 1 project\n"
            "Month 4: DSA — Graphs, DP, Greedy + Core subjects (OOP/DBMS/OS)\n"
            "Month 5: Build portfolio / 2nd project, 1 certification (try NPTEL / Coursera)\n"
            "Month 6: Mock interviews, 100+ aptitude questions/day, Resume finalization, Apply!\n\n"
            "Daily routine: 2hrs DSA + 1hr aptitude + 1hr project/skill + 30min English/news"
        )
    if category == 'career':
        return (
            "Choosing a career / domain:\n"
            "1. Don't chase salary alone — pick what you enjoy doing 8 hrs/day\n"
            "2. For CS/IT students, popular roles:\n"
            "   • Software Engineer (SDE) → strong coding + DSA needed\n"
            "   • Data Analyst → SQL + Excel + BI tools (Tableau/PowerBI)\n"
            "   • Data Scientist → Python + Stats + ML\n"
            "   • Full-stack Dev → React + Node/Express + DB\n"
            "   • QA / Testing → Selenium, API testing basics\n"
            "3. For MBA students: Marketing, Finance, HR, Operations, Analytics — choose based on interest + college major\n"
            "4. Try each role for 2 weeks with a mini-project — the one you don't get bored of → your answer!\n"
            "5. Network with 5 seniors/alumni in that role on LinkedIn — ask them \"a day in your life\" questions."
        )
    # category == 'placement' / generic
    return (
        "Placement drive checklist for you:\n"
        "✅ Resume updated to 1 page (PDF) — checked by 2 seniors\n"
        "✅ LinkedIn profile complete with photo, headline, projects\n"
        "✅ GitHub portfolio: at least 3 repos with clean README.md\n"
        "✅ 500+ aptitude questions practised (Quant + LR + Verbal)\n"
        "✅ 150+ DSA problems solved (Arrays → Trees → Graph → DP)\n"
        "✅ Core subjects revision notes (OOP, DBMS, OS, CN)\n"
        "✅ 2 projects prepared end-to-end for explanation\n"
        "✅ HR answers ready (Intro, Strengths, Weakness, Why company, Failures)\n"
        "✅ Formals / 2 copies of resume for the D-Day\n\n"
        "Ask me specifically — e.g. \"How do I prepare resume?\", \"Tips for HR interview?\""
    )

def local_chat_response(message):
    """Placement-aware local fallback — rejects off-topic, else category-specific answers."""
    ok, category = is_placement_related(message)
    if not ok:
        return OFF_TOPIC_RESPONSE
    return smart_local_chat_response(message, category)

# Chatbot route using Gemini API with strict placement guard
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = (request.get_json(silent=True) or {}).get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Please enter a message.'}), 400

    # Hard pre-filter: never even call the LLM for clearly off-topic questions
    ok, category = is_placement_related(user_message)
    if not ok:
        return jsonify({'response': OFF_TOPIC_RESPONSE})

    try:
        if not (genai and GEMINI_API_KEY):
            return jsonify({'response': smart_local_chat_response(user_message, category)})

        model = genai.GenerativeModel('gemini-2.0-flash')

        context = (
            "You are a Placement Assistant chatbot for an Indian college placement portal. "
            "Your ONLY job is to answer questions related to campus placements, job preparation, "
            "resume and cover letters, interviews (Technical / HR / GD / Aptitude), internships, "
            "career guidance for students, domain and role selection, academic eligibility (CGPA, "
            "10th/12th marks, backlogs, gap years), company recruitment processes, CTC and salary "
            "expectations, certifications, projects, portfolios, and skills to learn for placements.\n\n"

            "RULES YOU MUST STRICTLY FOLLOW:\n"
            "1. If the user asks ANYTHING outside placement / career prep — politely refuse and "
            "reply ONLY with the following paragraph unchanged:\n"
            "\"I specialize in placement guidance, resume/interview prep, and career planning for "
            "campus drives. I cannot help with unrelated topics. Please ask me about placements, "
            "resumes, interviews, aptitude, skills, projects, or career options.\"\n"
            "2. Keep answers structured (bulleted, numbered), practical, and specific to the "
            "Indian placement ecosystem (MNCs, mass recruiters, start-ups, Tier 1/2/3 colleges).\n"
            "3. Do NOT give medical, legal, financial-investment, dating, or political advice.\n"
            "4. Keep responses concise and under 150 words unless the user asks for depth.\n"
            "5. Be encouraging and realistic — no false promises about \"guaranteed placement\".\n"
            "6. If user's question is vague, ask a short clarifying question (e.g., \"Which role / "
            "company are you targeting so I can give precise tips?\")."
        )

        chat = model.start_chat(history=[])
        response = chat.send_message(f"{context}\n\nUser query: {user_message}")

        # Safety post-filter: if the model somehow ignored context, use the local fallback
        reply = (response.text or '').strip()
        return jsonify({'response': reply or smart_local_chat_response(user_message, category)})
    except Exception:
        return jsonify({'response': smart_local_chat_response(user_message, category)})

# Create tables during application setup
with app.app_context():
    db.create_all()

# Gunicorn / WSGI compatibility — expose the Flask app
application = app

# Start Flask server (local dev only)
if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', '0') == '1')

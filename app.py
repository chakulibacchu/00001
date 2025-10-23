from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import re
from datetime import datetime, timedelta
import time

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS for all origins

# Load Firebase config from environment variable
firebase_config_json = os.environ.get("FIREBASE_CONFIG")
if not firebase_config_json:
    raise EnvironmentError("FIREBASE_CONFIG environment variable not set")

try:
    firebase_json = json.loads(firebase_config_json)
except json.JSONDecodeError:
    raise ValueError("FIREBASE_CONFIG is not valid JSON")

# Initialize Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    initialize_app(cred)

# Firestore client
db = firestore.client()

def save_to_firebase(user_id, category, doc_id, data):
    """
    Save a document under users/{user_id}/{category}/{doc_id}.
    """
    if not user_id:
        return
    try:
        doc_ref = db.collection("users").document(user_id).collection(category).document(doc_id)
        doc_ref.set(data)
    except Exception as e:
        print(f"[FIREBASE ERROR] {e}")


client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LOGS_FILE = "logs.json"
REWARD_FILE = "user_rewards.json"

def load_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def read_logs():
    if not os.path.exists(LOGS_FILE):
        return []
    with open(LOGS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def write_logs(logs):
    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def read_rewards():
    if not os.path.exists(REWARD_FILE):
        return {}
    with open(REWARD_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def write_rewards(data):
    with open(REWARD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



# ============================================
# GENUINE APPRECIATION SKILL ENDPOINTS
# ============================================

@app.route('/api/skills/chat/message', methods=['POST', 'OPTIONS'])
def handle_chat_message():
    """Handle AI coach chat interaction"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    session_id = data.get("session_id")
    message = data.get("message", "").strip()
    chat_step = data.get("chat_step", 0)
    skill_id = data.get("skill_id", "genuine_appreciation")
    
    # Validation
    if not user_id or not message:
        return jsonify({"error": "Missing required fields: user_id, message"}), 400
    
    try:
        # Fetch user's condensed profile for personalization
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_data = user_doc.to_dict()
        condensed_profile = user_data.get("condensed_profile", "")
        
        # Load prompt template
        try:
            with open("prompt_appreciation_coach.txt", "r") as f:
                coach_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_appreciation_coach.txt not found"}), 500
        
        # Fetch chat history for context
        chat_history = []
        if session_id:
            chat_ref = db.collection("chat_sessions").document(session_id)
            chat_doc = chat_ref.get()
            if chat_doc.exists:
                chat_history = chat_doc.to_dict().get("messages", [])
        else:
            # Create new session
            session_id = db.collection("chat_sessions").document().id
        
        # Build conversation context
        conversation_context = "\n".join([
            f"{'User' if msg['type'] == 'user' else 'Coach'}: {msg['content']}"
            for msg in chat_history[-6:]  # Last 3 exchanges
        ])
        
        # Build system prompt
        system_prompt = coach_prompt_template.format(
            skill_id=skill_id,
            chat_step=chat_step,
            user_message=message,
            condensed_profile=condensed_profile,
            conversation_context=conversation_context
        )
        
        # Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Save chat history
        chat_history.extend([
            {"type": "user", "content": message, "timestamp": firestore.SERVER_TIMESTAMP},
            {"type": "bot", "content": ai_response, "timestamp": firestore.SERVER_TIMESTAMP}
        ])
        
        db.collection("chat_sessions").document(session_id).set({
            "user_id": user_id,
            "skill_id": skill_id,
            "messages": chat_history,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return jsonify({
            "response": ai_response,
            "session_id": session_id,
            "next_step": chat_step + 1
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/scenarios', methods=['GET', 'OPTIONS'])
def get_scenarios():
    """Get practice scenarios for a skill"""
    if request.method == 'OPTIONS':
        return '', 204
    
    skill_id = request.args.get("skill_id", "genuine_appreciation")
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400
    
    try:
        # Fetch scenarios from Firestore
        scenarios_ref = db.collection("scenarios").where("skill_id", "==", skill_id)
        scenarios_docs = scenarios_ref.stream()
        
        scenarios = []
        for doc in scenarios_docs:
            scenario_data = doc.to_dict()
            scenario_data["id"] = doc.id
            scenarios.append(scenario_data)
        
        # If no scenarios in DB, return hardcoded ones
        if not scenarios:
            scenarios = get_default_scenarios(skill_id)
        
        return jsonify({"scenarios": scenarios}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/scenarios/evaluate', methods=['POST', 'OPTIONS'])
def evaluate_scenario():
    """Evaluate user's scenario response"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    scenario_id = data.get("scenario_id")
    selected_option_id = data.get("selected_option_id")
    skill_id = data.get("skill_id", "genuine_appreciation")
    
    # Validation
    if not user_id or not scenario_id or not selected_option_id:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        # Fetch scenario from Firestore
        scenario_doc = db.collection("scenarios").document(scenario_id).get()
        
        if not scenario_doc.exists:
            return jsonify({"error": "Scenario not found"}), 404
        
        scenario_data = scenario_doc.to_dict()
        
        # Find selected option
        selected_option = None
        for option in scenario_data.get("options", []):
            if option["id"] == selected_option_id:
                selected_option = option
                break
        
        if not selected_option:
            return jsonify({"error": "Option not found"}), 404
        
        # Get user's current progress
        user_progress_ref = db.collection("users").document(user_id).collection("skill_progress").document(skill_id)
        progress_doc = user_progress_ref.get()
        
        current_xp = 0
        completed_scenarios = []
        
        if progress_doc.exists:
            progress_data = progress_doc.to_dict()
            current_xp = progress_data.get("total_xp", 0)
            completed_scenarios = progress_data.get("completed_scenarios", [])
        
        # Update XP
        new_xp = current_xp + selected_option.get("xp", 0)
        
        # Add scenario to completed list if not already there
        if scenario_id not in completed_scenarios:
            completed_scenarios.append(scenario_id)
        
        # Save progress
        user_progress_ref.set({
            "skill_id": skill_id,
            "total_xp": new_xp,
            "completed_scenarios": completed_scenarios,
            "last_scenario_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return jsonify({
            "is_correct": selected_option.get("isGenuine", False),
            "feedback": selected_option.get("feedback", ""),
            "xp_earned": selected_option.get("xp", 0),
            "total_xp": new_xp,
            "is_complete": len(completed_scenarios) >= get_total_scenarios_count(skill_id)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/progress', methods=['POST', 'OPTIONS'])
def save_progress():
    """Save user progress through skill"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    skill_id = data.get("skill_id", "genuine_appreciation")
    stage = data.get("stage")
    completed_scenarios = data.get("completed_scenarios", [])
    total_xp = data.get("total_xp", 0)
    reflection = data.get("reflection", "")
    mission_target = data.get("mission_target", "")
    
    # Validation
    if not user_id or not stage:
        return jsonify({"error": "Missing required fields: user_id, stage"}), 400
    
    try:
        # Save to Firestore
        progress_ref = db.collection("users").document(user_id).collection("skill_progress").document(skill_id)
        
        progress_data = {
            "skill_id": skill_id,
            "stage": stage,
            "completed_scenarios": completed_scenarios,
            "total_xp": total_xp,
            "updated_at": firestore.SERVER_TIMESTAMP
        }
        
        if reflection:
            progress_data["reflection"] = reflection
            progress_data["reflection_saved_at"] = firestore.SERVER_TIMESTAMP
        
        if mission_target:
            progress_data["mission_target"] = mission_target
            progress_data["mission_created_at"] = firestore.SERVER_TIMESTAMP
            progress_data["mission_deadline"] = firestore.SERVER_TIMESTAMP  # Add 24 hours
            progress_data["mission_status"] = "active"
        
        progress_ref.set(progress_data, merge=True)
        
        # If skill is completed, update user's main document
        if stage == "complete":
            db.collection("users").document(user_id).set({
                "completed_skills": firestore.ArrayUnion([skill_id]),
                "total_xp": firestore.Increment(total_xp),
                "last_skill_completed": skill_id,
                "last_skill_completed_at": firestore.SERVER_TIMESTAMP
            }, merge=True)
        
        return jsonify({
            "success": True,
            "progress_id": skill_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/progress', methods=['GET', 'OPTIONS'])
def get_progress():
    """Get user's progress for a skill"""
    if request.method == 'OPTIONS':
        return '', 204
    
    user_id = request.args.get("user_id")
    skill_id = request.args.get("skill_id", "genuine_appreciation")
    
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400
    
    try:
        progress_ref = db.collection("users").document(user_id).collection("skill_progress").document(skill_id)
        progress_doc = progress_ref.get()
        
        if not progress_doc.exists:
            return jsonify({
                "stage": "intro",
                "total_xp": 0,
                "completed_scenarios": [],
                "exists": False
            }), 200
        
        progress_data = progress_doc.to_dict()
        progress_data["exists"] = True
        
        return jsonify(progress_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/reflection', methods=['POST', 'OPTIONS'])
def save_reflection():
    """Save user's personal reflection"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    skill_id = data.get("skill_id", "genuine_appreciation")
    reflection = data.get("reflection", "").strip()
    
    # Validation
    if not user_id or not reflection:
        return jsonify({"error": "Missing required fields: user_id, reflection"}), 400
    
    if len(reflection) < 50:
        return jsonify({"error": "Reflection too short. Please write at least 50 characters."}), 400
    
    try:
        reflection_ref = db.collection("reflections").document()
        reflection_id = reflection_ref.id
        
        reflection_ref.set({
            "id": reflection_id,
            "user_id": user_id,
            "skill_id": skill_id,
            "reflection": reflection,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        # Also save to user's progress
        db.collection("users").document(user_id).collection("skill_progress").document(skill_id).set({
            "reflection": reflection,
            "reflection_saved_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return jsonify({
            "success": True,
            "reflection_id": reflection_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/mission', methods=['POST', 'OPTIONS'])
def save_mission():
    """Save user's mission"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    skill_id = data.get("skill_id", "genuine_appreciation")
    mission_target = data.get("mission_target", "").strip()
    
    # Validation
    if not user_id or not mission_target:
        return jsonify({"error": "Missing required fields: user_id, mission_target"}), 400
    
    try:
        from datetime import datetime, timedelta
        
        mission_ref = db.collection("missions").document()
        mission_id = mission_ref.id
        
        deadline = datetime.now() + timedelta(hours=24)
        
        mission_ref.set({
            "id": mission_id,
            "user_id": user_id,
            "skill_id": skill_id,
            "mission_target": mission_target,
            "status": "active",
            "deadline": deadline,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        # Also save to user's progress
        db.collection("users").document(user_id).collection("skill_progress").document(skill_id).set({
            "mission_target": mission_target,
            "mission_id": mission_id,
            "mission_status": "active",
            "mission_created_at": firestore.SERVER_TIMESTAMP,
            "mission_deadline": deadline
        }, merge=True)
        
        return jsonify({
            "success": True,
            "mission_id": mission_id,
            "deadline": deadline.isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/mission/<mission_id>/complete', methods=['PUT', 'OPTIONS'])
def complete_mission(mission_id):
    """Mark mission as completed"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    feedback = data.get("feedback", "")
    
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400
    
    try:
        mission_ref = db.collection("missions").document(mission_id)
        mission_doc = mission_ref.get()
        
        if not mission_doc.exists:
            return jsonify({"error": "Mission not found"}), 404
        
        mission_data = mission_doc.to_dict()
        
        if mission_data.get("user_id") != user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        # Update mission
        mission_ref.update({
            "status": "completed",
            "completed_at": firestore.SERVER_TIMESTAMP,
            "feedback": feedback
        })
        
        # Bonus XP for completing mission
        bonus_xp = 10
        skill_id = mission_data.get("skill_id")
        
        db.collection("users").document(user_id).collection("skill_progress").document(skill_id).update({
            "mission_status": "completed",
            "mission_completed_at": firestore.SERVER_TIMESTAMP,
            "total_xp": firestore.Increment(bonus_xp)
        })
        
        return jsonify({
            "success": True,
            "bonus_xp": bonus_xp
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/skills/stats', methods=['GET', 'OPTIONS'])
def get_user_stats():
    """Get overall user statistics"""
    if request.method == 'OPTIONS':
        return '', 204
    
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400
    
    try:
        user_doc = db.collection("users").document(user_id).get()
        
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        user_data = user_doc.to_dict()
        
        # Get all skill progress
        skills_ref = db.collection("users").document(user_id).collection("skill_progress")
        skills_docs = skills_ref.stream()
        
        total_xp = 0
        completed_skills = []
        active_skills = []
        
        for skill_doc in skills_docs:
            skill_data = skill_doc.to_dict()
            skill_id = skill_doc.id
            
            total_xp += skill_data.get("total_xp", 0)
            
            if skill_data.get("stage") == "complete":
                completed_skills.append(skill_id)
            else:
                active_skills.append(skill_id)
        
        return jsonify({
            "total_xp": total_xp,
            "completed_skills": completed_skills,
            "active_skills": active_skills,
            "user_name": user_data.get("name", "User")
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_default_scenarios(skill_id):
    """Return hardcoded scenarios if none in database"""
    if skill_id == "genuine_appreciation":
        return [
            {
                "id": "scenario1",
                "skill_id": "genuine_appreciation",
                "context": "workplace",
                "situation": "Your coworker Sarah stayed late last night to help you meet a tight deadline. The project was successful.",
                "character": "👩‍💼",
                "options": [
                    {
                        "id": "opt1",
                        "text": "Thanks for your help!",
                        "isGenuine": False,
                        "feedback": "This is polite but generic. It doesn't acknowledge what Sarah specifically did or the impact it had.",
                        "xp": 5
                    },
                    {
                        "id": "opt2",
                        "text": "Sarah, I really appreciate you staying late to help with the analytics section. Your attention to detail caught errors I completely missed, and it made our presentation so much stronger.",
                        "isGenuine": True,
                        "feedback": "Perfect! This is genuine because it's specific (analytics section), acknowledges the sacrifice (staying late), recognizes a quality (attention to detail), and explains the impact (stronger presentation).",
                        "xp": 25
                    },
                    {
                        "id": "opt3",
                        "text": "You're such a team player, Sarah!",
                        "isGenuine": False,
                        "feedback": "While positive, this is a surface-level compliment. It doesn't reference specific actions or show you truly noticed what she did.",
                        "xp": 10
                    }
                ]
            },
            {
                "id": "scenario2",
                "skill_id": "genuine_appreciation",
                "context": "personal",
                "situation": "Your friend remembered your job interview and texted you asking how it went, even though they were dealing with their own stressful week.",
                "character": "🧑‍🤝‍🧑",
                "options": [
                    {
                        "id": "opt1",
                        "text": "Thanks for checking in!",
                        "isGenuine": False,
                        "feedback": "Too brief. Doesn't acknowledge that they made time despite their own stress.",
                        "xp": 5
                    },
                    {
                        "id": "opt2",
                        "text": "I really appreciate that you remembered and reached out, especially knowing you've had a tough week yourself. It means a lot that you made space to care about what's going on with me.",
                        "isGenuine": True,
                        "feedback": "Excellent! You acknowledged their specific action (reaching out), recognized their context (tough week), and explained the emotional impact (means a lot).",
                        "xp": 25
                    },
                    {
                        "id": "opt3",
                        "text": "You're the best friend ever!",
                        "isGenuine": False,
                        "feedback": "Hyperbolic and vague. Genuine appreciation is about specific observations, not generic superlatives.",
                        "xp": 10
                    }
                ]
            }
        ]
    return []

def get_total_scenarios_count(skill_id):
    """Get total number of scenarios for a skill"""
    scenarios_ref = db.collection("scenarios").where("skill_id", "==", skill_id)
    return len(list(scenarios_ref.stream()))
    

@app.route('/api/generate-briefing', methods=['POST', 'OPTIONS'])
def generate_briefing():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    location = data.get("location", "").strip()
    time = data.get("time", "").strip()
    energy_level = data.get("energy_level", 3)
    confidence_level = data.get("confidence_level", 3)
    user_history = data.get("user_history", {})
    
    # Validation
    if not user_id or not location or not time:
        return jsonify({"error": "Missing required fields: user_id, location, time"}), 400
    
    try:
        # Fetch user's condensed profile for personalization
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        condensed_profile = user_doc.to_dict().get("condensed_profile", "")
        
        # Load prompt template
        try:
            with open("prompt_mission_briefing.txt", "r") as f:
                briefing_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_mission_briefing.txt not found"}), 500
        
        # Build system prompt with user context
        system_prompt = briefing_prompt_template.format(
            location=location,
            time=time,
            energy_level=energy_level,
            confidence_level=confidence_level,
            condensed_profile=condensed_profile,
            user_history=json.dumps(user_history)
        )
        
        # Call LLM to generate briefing
        messages = [{"role": "system", "content": system_prompt}]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        briefing_text = response.choices[0].message.content.strip()
        
        # Parse the response into structured format
        briefing_data = parse_briefing_response(briefing_text)
        
        # Save briefing to user's Firestore document
        db.collection("users").document(user_id).set(
            {
                "last_briefing": {
                    "location": location,
                    "time": time,
                    "energy_level": energy_level,
                    "confidence_level": confidence_level,
                    "briefing_data": briefing_data,
                    "created_at": firestore.SERVER_TIMESTAMP
                }
            },
            merge=True
        )
        
        return jsonify(briefing_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINT 2: Regenerate Openers Only
# ============================================================================

@app.route('/api/regenerate-openers', methods=['POST', 'OPTIONS'])
def regenerate_openers():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    location = data.get("location", "").strip()
    confidence_level = data.get("confidence_level", 3)
    previous_openers = data.get("previous_openers", [])
    
    if not user_id or not location:
        return jsonify({"error": "Missing required fields: user_id, location"}), 400
    
    try:
        # Fetch user profile
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        condensed_profile = user_doc.to_dict().get("condensed_profile", "")
        
        # Load openers prompt
        try:
            with open("prompt_openers.txt", "r") as f:
                openers_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_openers.txt not found"}), 500
        
        system_prompt = openers_prompt_template.format(
            location=location,
            confidence_level=confidence_level,
            condensed_profile=condensed_profile,
            previous_opener_ids=",".join(previous_openers)
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.8,
            max_tokens=1200
        )
        
        openers_text = response.choices[0].message.content.strip()
        openers = parse_openers_response(openers_text)
        
        return jsonify({"openers": openers}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINT 3: Save Favorite Opener
# ============================================================================

@app.route('/api/save-favorite-opener', methods=['POST', 'OPTIONS'])
def save_favorite_opener():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    opener_id = data.get("opener_id")
    
    if not user_id or not opener_id:
        return jsonify({"error": "Missing required fields: user_id, opener_id"}), 400
    
    try:
        # Add opener to user's favorite_openers array
        db.collection("users").document(user_id).set(
            {
                "favorite_openers": firestore.ArrayUnion([opener_id]),
                "last_favorite_saved": firestore.SERVER_TIMESTAMP
            },
            merge=True
        )
        
        return jsonify({
            "success": True,
            "message": "Opener saved to favorites"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS: Parsing LLM Responses
# ============================================================================

def parse_briefing_response(text):
    """
    Parse the LLM response into structured briefing data.
    The prompt should instruct the LLM to return JSON.
    """
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback: return raw text in a structured format
            return {
                "venue_intel": {"raw_analysis": text},
                "openers": [],
                "scenarios": [],
                "conversation_flows": [],
                "cheat_sheet": text
            }
    except Exception as e:
        return {
            "error": "Failed to parse briefing",
            "raw_response": text
        }


def parse_openers_response(text):
    """
    Parse opener data from LLM response into structured format.
    """
    try:
        import re
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback: return empty list
            return []
    except Exception as e:
        return []



# ============================================================================
# OPTIONAL: Save Briefing Session for Analytics
# ============================================================================

@app.route('/api/save-briefing-session', methods=['POST', 'OPTIONS'])
def save_briefing_session():
    """
    Save user's briefing session for future learning and improvement.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    session_data = data.get("session_data")  # outcomes, what worked, etc.
    
    if not user_id or not session_data:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        db.collection("users").document(user_id).collection("briefing_history").add({
            "session_data": session_data,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "success": True,
            "message": "Session saved for future insights"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "✅ Groq LLaMA 4 Scout Backend is running."

@app.route('/anxiety-chat', methods=['POST', 'OPTIONS'])
def anxiety_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    try:
        data = request.get_json()
        user_id = data.get("user_id")
        conversation_id = data.get("conversation_id")
        message_type = data.get("message_type")
        context = data.get("context", {})
        user_input = context.get("user_input", "")
        
        if not user_id or not conversation_id or not message_type:
            return jsonify({"error": "Missing required fields"}), 400

        # Get API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer "):].strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401

        # Initialize client with provided API key
        client.api_key = api_key

        # Load conversation history from Firebase
        doc_ref = db.collection("anxiety_conversations").document(conversation_id)
        doc = doc_ref.get()

        if doc.exists:
            history = doc.to_dict().get("messages", [])
        else:
            # First time: load the anxiety reduction prompt
            try:
                with open("prompt_anxiety_reduction.txt", "r") as f:
                    system_prompt = f.read()
            except FileNotFoundError:
                return jsonify({"error": "prompt_anxiety_reduction.txt not found"}), 500
            
            history = [{"role": "system", "content": system_prompt}]

        # Build context-aware message based on message_type
        if message_type == "greeting":
            user_message = f"I'm about to have a {context.get('task', {}).get('type', 'social')} interaction. I'm feeling anxious."
        
        elif message_type == "exercise_recommendation":
            user_state = context.get('user_state', {})
            user_message = f"""Based on my current state:
- Anxiety level: {user_state.get('anxietyLevel', 3)}/5
- Energy level: {user_state.get('energyLevel', 3)}/5
- Main worry: {user_state.get('worry', 'unknown')}
- Interaction type: {context.get('task', {}).get('type', 'unknown')}

What exercises should I do to prepare? Respond with a supportive message and suggest exercises from: grounding, breathing, ai-chat, self-talk, physical."""
        
        elif message_type == "motivation":
            exercises_completed = context.get('exercise_history', [])
            user_message = f"I just completed {len(exercises_completed)} exercise(s): {', '.join(exercises_completed)}. Give me encouraging feedback!"
        
        elif message_type == "self_talk_generation":
            user_state = context.get('user_state', {})
            user_message = f"""Generate 4 personalized positive affirmations for someone who:
- Has anxiety level {user_state.get('anxietyLevel', 3)}/5
- Main worry: {user_state.get('worry', 'unknown')}
- About to have a {context.get('task', {}).get('type', 'social')} interaction

Format: Return ONLY a JSON array of 4 strings, nothing else."""
        
        elif message_type == "reflection_prompt":
            user_message = "I've completed my preparation exercises. Help me reflect on what I accomplished."
        
        elif message_type == "reflection_analysis":
            reflection = context.get('reflection', {})
            user_message = f"""I just reflected on my preparation:
- Anxiety before: {context.get('user_state', {}).get('anxietyLevel', 3)}/5
- Anxiety after: {reflection.get('finalAnxiety', 3)}/5
- Confidence: {reflection.get('finalConfidence', 3)}/5
- Exercises helped: {reflection.get('exercisesHelped', 'unknown')}

Give me encouraging analysis of my progress!"""
        
        elif message_type == "emergency_followup":
            user_message = "I just did a 60-second emergency breathing reset. Check in on me."
        
        elif message_type == "user_message":
            user_message = user_input
        
        else:
            user_message = user_input or "Help me with my anxiety."

        # Append user message to history
        history.append({"role": "user", "content": user_message})

        # Call the AI model
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.7 if message_type == "user_message" else 0.6,
            max_tokens=500 if message_type == "user_message" else 300
        )

        ai_reply = response.choices[0].message.content.strip()

        # Handle self-talk generation specially (extract JSON)
        suggestions = None
        if message_type == "self_talk_generation":
            try:
                import json
                # Try to extract JSON array from response
                if "[" in ai_reply and "]" in ai_reply:
                    json_start = ai_reply.index("[")
                    json_end = ai_reply.rindex("]") + 1
                    suggestions = json.loads(ai_reply[json_start:json_end])
                else:
                    # Fallback: split by newlines or bullets
                    suggestions = [line.strip("- •") for line in ai_reply.split("\n") if line.strip()][:4]
            except:
                suggestions = [
                    "I am capable and prepared.",
                    "It's okay to feel nervous.",
                    "I've handled situations like this before.",
                    "One step at a time is enough."
                ]

        # Append AI response to history
        history.append({"role": "assistant", "content": ai_reply})

        # Save updated conversation to Firebase
        doc_ref.set({
            "messages": history,
            "user_id": user_id,
            "last_updated": firestore.SERVER_TIMESTAMP
        }, merge=True)

        # Return response
        response_data = {"response": ai_reply}
        if suggestions:
            response_data["suggestions"] = suggestions

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/reply-day-chat-advanced', methods=['POST', 'OPTIONS'])
def reply_day_chat_advanced():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message", "").strip()
    goal_name = data.get("goal_name", "").strip()
    user_places = data.get("user_places", [])
    user_interests = data.get("user_interests", [])

    if not user_id or not message:
        return jsonify({"error": "Missing input"}), 400

    # Fetch latest chat for the day
    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    if not docs:
        return jsonify({"error": "Chat not started"}), 404

    doc_ref = docs[0].reference
    chat_data = docs[0].to_dict()
    chat_history = chat_data.get("chat", [])

    # Append user message
    chat_history.append({"role": "user", "content": message})

    # Load chat prompt
    try:
        with open("prompt_DAYONE_COMPONENTONE.txt", "r") as f:
            chat_prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt_DAYONE_COMPONENTONE not found"}), 500

    # Inject user-specific info into the prompt
    system_prompt = chat_prompt_template.format(
        goal_name=goal_name or "their personal goal",
        user_places=", ".join(user_places) if user_places else "none",
        user_interests=", ".join(user_interests) if user_interests else "none"
    )

    context_message = {"role": "system", "content": system_prompt}
    messages_for_model = [chat_history[0]] + [context_message] + chat_history[1:]

    try:
        # Generate AI chat reply
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages_for_model,
            temperature=0.6,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()

        # Append AI response
        chat_history.append({"role": "assistant", "content": reply})
        doc_ref.update({"chat": chat_history})

        # ----------------------
        # Generate condensed profile
        # ----------------------
        condensed_prompt = f"""
        You are an assistant that builds a concise user profile.
        Based on the following conversation, summarize the user's:
        - Places they visit regularly
        - Social habits and interactions
        - Interests and hobbies
        - Any personality/behavioral cues
        Only output a structured JSON with keys: places, social_habits, interests, personality.

        Conversation:
        {chat_history}
        """

        profile_response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": condensed_prompt}],
            temperature=0.3,
            max_tokens=300
        )

        condensed_profile_text = profile_response.choices[0].message.content.strip()

        # Save condensed profile
        db.collection("users").document(user_id).set(
            {"condensed_profile": condensed_profile_text},
            merge=True
        )

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-user-places', methods=['POST', 'OPTIONS'])
def generate_user_places():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    goal_name = data.get("goal_name", "").strip()

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # Fetch condensed profile
    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        return jsonify({"error": "User not found or profile not generated yet"}), 404

    condensed_profile = user_doc.to_dict().get("condensed_profile", "")
    if not condensed_profile:
        return jsonify({"error": "Condensed profile is empty"}), 404

    # Load location prompt
    try:
        with open("prompt_location.txt", "r") as f:
            location_prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt_location.txt not found"}), 500

    # Inject user info into location prompt
    system_prompt = location_prompt_template.format(
        goal_name=goal_name or "their personal goal",
        condensed_profile=condensed_profile
    )

    messages_for_model = [{"role": "system", "content": system_prompt}]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages_for_model,
            temperature=0.6,
            max_tokens=400
        )
        suggested_places = response.choices[0].message.content.strip()

        # Optionally save suggested places back to user doc
        db.collection("users").document(user_id).set(
            {"suggested_places": suggested_places},
            merge=True
        )

        return jsonify({"suggested_places": suggested_places})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        user_message = data.get("message", "").strip()
        goal_name = data.get("goal_name", "").strip()

        if not user_id or not user_message:
            return jsonify({"error": "Missing user_id or message"}), 400

        # Get API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer "):].strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401

        # Initialize client with provided API key
        client.api_key = api_key

        # Load conversation history from Firebase
        doc_ref = db.collection("conversations").document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            history = doc.to_dict().get("messages", [])
        else:
            # First time: load your prompt file as the system instruction
            prompt_template = load_prompt("prompt_setgoal.txt")
            if not prompt_template:
                return jsonify({"error": "prompt_setgoal.txt not found"}), 500

            # Inject goal_name into the prompt
            system_prompt = prompt_template.format(goal_name=goal_name or "their personal goal")
            history = [{"role": "system", "content": system_prompt}]

        # Always reinforce goal_name context
        context_message = {
            "role": "system",
            "content": f"Reminder: the user’s goal/context is '{goal_name or 'their personal goal'}'. "
                       f"Keep this in mind when responding."
        }

        # Build full message list for the AI
        messages_for_model = [history[0], context_message] + history[1:]
        messages_for_model.append({"role": "user", "content": user_message})

        # Call the LLaMA / Groq model
        response = client.chat.completions.create(
            model="groq/compound",
            messages=messages_for_model,
            temperature=0.7,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()

        # Check if AI provided a finalized goal (expects "Final Goal: <goal_name>" in reply)
        final_goal = None
        if "Final Goal:" in ai_message:
            parts = ai_message.split("Final Goal:")
            ai_message = parts[0].strip()  # keep conversational reply
            final_goal = parts[1].strip()

        # Append user + AI message to history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_message})

        # Save updated conversation to Firebase
        doc_ref.set({"messages": history})

        # Return reply + optional finalized goal
        return jsonify({
            "reply": ai_message,
            "final_goal": final_goal
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500




@app.route("/mindpal-reward", methods=["POST"])
def mindpal_reward_webhook():
    data = request.get_json()
    user_id = data.get("user_id")
    rewards = data.get("rewards", [])

    if not user_id or not isinstance(rewards, list):
        return jsonify({"error": "Missing user_id or rewards[]"}), 400

    # 🔥 Save to: users/<user_id>/rewards/<auto_id>
    save_to_firebase(user_id, "rewards", {
        "source": "mindpal",
        "rewards": rewards
    })

    # ✅ Optionally also save to local file (if still needed)
    local_data = read_rewards()
    local_data[user_id] = {
        "reward_list": rewards,
        "source": "mindpal"
    }
    write_rewards(local_data)

    return jsonify({"status": "Reward saved successfully"}), 200





@app.route('/create-dated-course', methods=['POST'])
def create_dated_course():
    data = request.get_json()
    print("📥 Received payload:", data)  # Log incoming request

    user_id = data.get("user_id")
    final_plan = data.get("final_plan")
    join_date_str = data.get("join_date")  # Optional: user join date

    if not user_id or not final_plan:
        print("❌ Missing required data")
        return jsonify({"error": "Missing required data"}), 400

    # Parse join date
    try:
        joined_date = datetime.strptime(join_date_str, "%Y-%m-%d") if join_date_str else datetime.now()
        print("📅 Parsed join date:", joined_date)
    except Exception as e:
        print("⚠️ Failed to parse join date, using current date. Error:", e)
        joined_date = datetime.now()

    # Convert final_plan into a dated plan
    dated_plan = {}
    for i, day_key in enumerate(final_plan.get("final_plan", {}), start=0):
        date_str = (joined_date + timedelta(days=i)).strftime("%Y-%m-%d")
        day_data = final_plan["final_plan"][day_key].copy()

        # Convert tasks into toggle-ready objects
        tasks_with_toggle = [{"task": t, "done": False} for t in day_data.get("tasks", [])]
        day_data["tasks"] = tasks_with_toggle

        dated_plan[date_str] = day_data

    print("📝 Dated plan prepared:", dated_plan)

    # Save to Firebase
    try:
        course_id = "social_skills_101"  # You can make this dynamic
        doc_path = f"dated_courses/{user_id}/{course_id}"
        print("📌 Writing to Firestore at:", doc_path)

        db.document(doc_path).set({
            "joined_date": joined_date.strftime("%Y-%m-%d"),
            "lessons_by_date": dated_plan
        })

        print("✅ Write successful")
        return jsonify({"success": True, "dated_plan": dated_plan})

    except Exception as e:
        print("❌ Failed to write to Firestore:", e)
        return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500



@app.route('/toggle-task', methods=['POST'])
def toggle_task():
    data = request.get_json()
    user_id = data.get("user_id")
    day = data.get("day")
    task_index = data.get("task_index")
    completed = data.get("completed")

    if user_id is None or day is None or task_index is None or completed is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Reference to user's task document for the day
    task_doc_ref = db.collection("users").document(user_id).collection("task_status").document(f"day_{day}")
    task_doc = task_doc_ref.get()

    if task_doc.exists:
        task_data = task_doc.to_dict()
        tasks_completed = task_data.get("tasks_completed", [])
    else:
        # Initialize if not exists
        tasks_completed = []

    # Ensure the tasks_completed array has enough slots
    while len(tasks_completed) <= task_index:
        tasks_completed.append(False)

    # Update the specific task's completion
    tasks_completed[task_index] = completed

    # Save back to Firestore
    task_doc_ref.set({
        "tasks_completed": tasks_completed,
        "timestamp": datetime.utcnow()
    })

    # Calculate daily progress
    total_tasks = len(tasks_completed)
    completed_count = sum(1 for t in tasks_completed if t)
    daily_progress = completed_count / total_tasks if total_tasks > 0 else 0

    return jsonify({
        "day": day,
        "task_index": task_index,
        "completed": completed,
        "daily_progress": daily_progress,
        "tasks_completed": tasks_completed
    })

if __name__ == "__main__":
    app.run(debug=True)


@app.route('/support-room-question', methods=['POST'])
def support_room_question():
    data = request.get_json()
    user_id = data.get("user_id")
    task = data.get("task", "").strip()
    question = data.get("question", "").strip()

    if not task or not question:
        return jsonify({"error": "Missing task or question"}), 400

    prompt_template = load_prompt("prompt_support_room.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_support_room.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<task>>", task)
        .replace("<<question>>", question)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        result = response.choices[0].message.content.strip()

        # Optionally: save in Firestore
        save_to_firebase(user_id, "support_room_responses", {
            "task": task,
            "question": question,
            "response": result
        })

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/rescue-plan-chat-answers', methods=['POST'])
def rescue_plan_chat_answers():
    data = request.get_json()
    user_id = data.get("user_id")
    task = data.get("task")
    answers = data.get("answers")  # list of 7 answers

    # ✅ Basic validation
    if not user_id or not task or not answers or not isinstance(answers, list):
        return jsonify({"error": "Missing or invalid data"}), 400

    try:
        # ✅ Save to Firestore
        save_to_firebase(user_id, "rescue_chat_answers", {
            "task": task,
            "answers": answers
        })

        return jsonify({"status": "success", "message": "Answers saved ✅"}), 200

    except Exception as e:
        print("❌ Error saving rescue chat answers:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/generate-action-level-questions', methods=['POST'])
def generate_action_level_questions():
    data = request.get_json()
    user_id = data.get("user_id", "")

    prompt_template = load_prompt("prompt_action_level_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_action_level_questions.txt not found"}), 500

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.4,
            max_tokens=400
        )
        result = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse questions JSON", "raw": result}), 500

        save_to_firebase(user_id, "action_level_questions", {
            "questions": parsed.get("questions", [])
        })

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


@app.route('/rescue-plan-chat-start', methods=['POST'])
def rescue_plan_chat_start():
    data = request.get_json()
    task = data.get("task", "")
    user_id = data.get("user_id", "")

    if not task:
        return jsonify({"error": "Missing task"}), 400

    prompt_template = load_prompt("prompt_rescue_chat_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_rescue_chat_questions.txt not found"}), 500

    prompt = prompt_template.replace("<<task>>", task)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)

        save_to_firebase(user_id, "rescue_chat_questions", {
            "task": task,
            "questions": parsed.get("questions", [])
        })

        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-rescue-kit', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_rescue_kit():
    if request.method == "OPTIONS":
        # Preflight request for CORS
        return '', 200

    try:
        data = request.get_json()
        user_id = data.get("userId")  # ✅ match frontend key (camelCase)
        task = data.get("task", "")
        risks = data.get("risks", [])  # list of strings
        reward = data.get("reward", "")  # optional

        if not task or not risks:
            return jsonify({"error": "Missing task or risks"}), 400

        risks_formatted = "\n".join([f"- {r}" for r in risks])

        prompt_template = load_prompt("prompt_rescue_kit.txt")
        if not prompt_template:
            return jsonify({"error": "prompt_rescue_kit.txt not found"}), 500

        prompt = (
            prompt_template
            .replace("<<task>>", task)
            .replace("<<risks>>", risks_formatted)
            .replace("<<reward>>", reward)
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=700
        )
        result = response.choices[0].message.content.strip()

        parsed = json.loads(result)

        save_to_firebase(user_id, "rescue_kit", {
            "task": task,
            "risks": risks,
            "reward": reward,
            "rescue_plans": parsed.get("plans", [])
        })

        return jsonify(parsed)
    
    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-action-level', methods=['POST'])
def analyze_action_level():
    data = request.get_json()
    user_id = data.get("user_id")
    answers = data.get("answers", [])

    if not user_id or not isinstance(answers, list) or not answers:
        return jsonify({"error": "Missing or invalid user_id or answers"}), 400

    formatted_answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

    prompt_template = load_prompt("prompt_analyze_action_level.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_analyze_action_level.txt not found"}), 500

    prompt = prompt_template.replace("<<userlevelanswers>>", formatted_answers)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        result = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse JSON", "raw_response": result}), 500

        # Store result in Firebase
        save_to_firebase(user_id, "action_level_analysis", {
            "answers": answers,
            "analysis": parsed
        })

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


@app.route('/achievement-summary', methods=['POST'])
def achievement_summary():
    data = request.get_json()
    user_id = data.get("user_id")
    plan = data.get("plan")  # The user's plan input (likely a dict)

    if not user_id or not plan:
        return jsonify({"error": "Missing user_id or plan"}), 400

    prompt_template = load_prompt("prompt_achievement_summary.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_achievement_summary.txt not found"}), 500

    # Inject the plan JSON into your prompt template
    prompt = prompt_template.replace("<<plan>>", json.dumps(plan, indent=2))

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600
        )
        achievement_text = response.choices[0].message.content.strip()

        # Optionally save achievement summary to Firebase
        save_to_firebase(user_id, "achievement_summaries", {
            "plan": plan,
            "achievement_summary": achievement_text
        })

        return jsonify({"achievement_summary": achievement_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/start-day-chat', methods=['POST', 'OPTIONS'])
def start_day_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    day_number = data.get("day_number")
    sections = data.get("subsections", [])

    if not user_id or not day_number or not isinstance(sections, list):
        return jsonify({"error": "Invalid input"}), 400

    prompt_template = load_prompt("prompt_customize_day.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_customize_day.txt not found"}), 500

    formatted_sections = "\n".join([f"- {s}" for s in sections])
    prompt = (
        prompt_template
        .replace("<<day_number>>", str(day_number))
        .replace("<<subsections>>", formatted_sections)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        msg = response.choices[0].message.content.strip()

        chat_data = {
            "day": day_number,
            "sections": sections,
            "chat": [{"role": "assistant", "content": msg}]
        }

        save_to_firebase(user_id, "custom_day_chat", chat_data)
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------- REPLY DAY CHAT ---------

@app.route('/reply-day-chat', methods=['POST', 'OPTIONS'])
def reply_day_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"error": "Missing input"}), 400

    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    if not docs:
        return jsonify({"error": "Chat not started"}), 404

    doc_ref = docs[0].reference
    chat_data = docs[0].to_dict()
    chat_history = chat_data.get("chat", [])

    chat_history.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=chat_history,
            temperature=0.5,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
        chat_history.append({"role": "assistant", "content": reply})

        doc_ref.update({"chat": chat_history})
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/finalize-day-chat', methods=['POST'])
def finalize_day_chat():
    data = request.get_json()
    user_id = data.get("user_id")
    user_data = data.get("user_data")
    ogplan = data.get("ogplan")

    if not user_id or not user_data or not ogplan:
        return jsonify({"error": "Missing required data"}), 400

    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    if not docs:
        return jsonify({"error": "No chat session found"}), 404

    chat = docs[0].to_dict()
    chat_history = chat.get("chat", [])
    day_number = chat.get("day")

    finalize_prompt = load_prompt("prompt_customize_day_finalize.txt")
    if not finalize_prompt:
        return jsonify({"error": "prompt_customize_day_finalize.txt not found"}), 500

    final_instruction = (
        finalize_prompt
        .replace("<<user_data>>", json.dumps(user_data, indent=2))
        .replace("<<ogplan>>", json.dumps(ogplan, indent=2))
        .replace("<<day_number>>", str(day_number))
    )

    chat_history.append({"role": "user", "content": final_instruction})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=chat_history,
            temperature=0.4,
            max_tokens=4000
        )
        final_output = response.choices[0].message.content.strip()

        # Remove ```json or ``` wrapping from the AI response
        cleaned_output = re.sub(r"^```(?:json)?|```$", "", final_output.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned_output)
        except json.JSONDecodeError as json_err:
            return jsonify({
                "error": "Failed to parse final JSON",
                "raw": final_output,
                "cleaned": cleaned_output,
                "details": str(json_err)
            }), 500

        final_data = {
            "day": day_number,
            "final_plan": parsed
        }

        save_to_firebase(user_id, "custom_day_final_plans", final_data)
        return jsonify({"final_plan": parsed})
    
    except Exception as e:
        return jsonify({"error": f"Backend error: {str(e)}"}), 500

@app.route("/get-ogplan", methods=["POST"])
def get_ogplan():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        plans = db.collection("users").document(user_id).collection("plans")
        docs = list(plans.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream())
        if not docs:
            return jsonify({"error": "No plan found"}), 404

        plan_data = docs[0].to_dict().get("ai_plan")
        return jsonify({"ogplan": plan_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-questions', methods=['POST'])
def ask_questions():
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()
    user_id = data.get("user_id")

    if not goal_name:
        return jsonify({"error": "Missing goal_name"}), 400

    prompt_template = load_prompt("prompt_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_questions.txt not found"}), 500

    prompt = prompt_template.format(goal_name=goal_name)

    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        result = response.choices[0].message.content.strip()
        save_to_firebase(user_id, "questions", {
            "goal_name": goal_name,
            "questions": result
        })
        return jsonify({"questions": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ============ HELPER FUNCTIONS ============

def load_prompt(filename):
    """Load prompt template from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def get_course_ref(user_id, course_id):
    """Get reference to the course document"""
    return db.collection('users').document(user_id).collection('datedcourses').document(course_id)

def determine_difficulty(task_text):
    """Determine task difficulty based on keywords"""
    lower_task = task_text.lower()
    if any(word in lower_task for word in ['review', 'reflect', 'schedule', 'take a few minutes', 'read']):
        return 'easy'
    elif any(word in lower_task for word in ['practice', 'connect', 'reach out', 'write', 'try']):
        return 'medium'
    else:
        return 'hard'

# ============ MAIN ENDPOINT CREATOR ============

def create_day_endpoint(day):
    endpoint_name = f"final_plan_day_{day}"
    route_path = f"/final-plan-day{day}"
    
    @app.route(route_path, methods=['POST'], endpoint=endpoint_name)
    def final_plan_day_func():
        # ========== STEP 1: Parse Request ==========
        data = request.get_json()
        goal_name = data.get("goal_name", "").strip()
        user_answers = data.get("user_answers", [])
        user_id = data.get("user_id", "").strip()
        join_date_str = data.get("join_date")
        
        if not goal_name or not isinstance(user_answers, list) or not user_id:
            return jsonify({"error": "Missing or invalid goal_name, user_answers, or user_id"}), 400
        
        # Parse join date
        try:
            joined_date = datetime.strptime(join_date_str, "%Y-%m-%d") if join_date_str else datetime.now()
        except:
            joined_date = datetime.now()
        
        # Calculate the date for this day
        day_date = (joined_date + timedelta(days=day-1)).strftime("%Y-%m-%d")
        course_id = goal_name.lower().replace(" ", "_")
        
        formatted_answers = "\n".join(
            [f"{i+1}. {answer.strip()}" for i, answer in enumerate(user_answers) if isinstance(answer, str)]
        )
        
        # Get API key from Authorization header
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401
        
        client.api_key = api_key
        
        # ========== STEP 2: Load Previous Day (if needed) ==========
        previous_day_lesson = None
        if day > 1:
            try:
                course_ref = get_course_ref(user_id, course_id)
                course_doc = course_ref.get()
                
                if course_doc.exists:
                    course_data = course_doc.to_dict()
                    lessons_by_date = course_data.get('lessons_by_date', {})
                    
                    # Get previous day's date
                    prev_day_date = (joined_date + timedelta(days=day-2)).strftime("%Y-%m-%d")
                    previous_day_lesson = lessons_by_date.get(prev_day_date)
                    
                    print(f"✅ Loaded previous day ({prev_day_date}) for context")
            except Exception as e:
                print(f"⚠️ Could not load previous day: {e}")
                previous_day_lesson = None
        
        # ========== STEP 3: Load Prompt Template ==========
        prompt_file = f"prompt_plan_{day:02}.txt"
        prompt_template = load_prompt(prompt_file)
        if not prompt_template:
            return jsonify({"error": f"{prompt_file} not found"}), 404
        
        # Replace placeholders
        prompt = prompt_template.replace("<<goal_name>>", goal_name).replace("<<user_answers>>", formatted_answers)
        if previous_day_lesson:
            prompt = prompt.replace(f"<<day_{day-1}_json>>", json.dumps(previous_day_lesson))
        
        # ========== STEP 4: Generate AI Plan ==========
        try:
            response = client.chat.completions.create(
                model="groq/compound",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=4096
            )
            result = response.choices[0].message.content.strip()
            parsed_day_plan = json.loads(result)
            print(f"✅ Day {day} plan generated from AI")
        except json.JSONDecodeError:
            return jsonify({"error": f"Failed to parse Day {day} as JSON", "raw_response": result}), 500
        except Exception as e:
            return jsonify({"error": f"API request failed", "exception": str(e)}), 500
        
        # ========== STEP 5: Transform to App Structure ==========
        lesson_data = {
            "title": parsed_day_plan.get("title", f"Day {day} Challenge"),
            "summary": parsed_day_plan.get("summary", ""),
            "lesson": parsed_day_plan.get("lesson", ""),
            "motivation": parsed_day_plan.get("motivation", ""),
            "why": parsed_day_plan.get("why", ""),
            "quote": parsed_day_plan.get("quote", ""),
            "consequences": parsed_day_plan.get("consequences", {
                "positive": "",
                "negative": ""
            }),
            "duration": parsed_day_plan.get("duration", "15 min"),
            "xp": parsed_day_plan.get("xp", 100),
            "date": day_date,
            "completed": False,
            "reflection": ""
        }
        
        # Convert tasks to app format
        raw_tasks = parsed_day_plan.get("tasks", [])
        if isinstance(raw_tasks, list):
            lesson_data["tasks"] = [
                {
                    "task": task if isinstance(task, str) else task.get("task", ""),
                    "done": False,
                    "difficulty": determine_difficulty(task if isinstance(task, str) else task.get("task", "")),
                    "timeSpent": 0,
                    "notes": ""
                }
                for task in raw_tasks
            ]
        else:
            lesson_data["tasks"] = []
        
        # Add quiz if present
        if "quiz" in parsed_day_plan:
            lesson_data["quiz"] = parsed_day_plan["quiz"]
        
        print(f"✅ Day {day} lesson structured for date: {day_date}")
        
        # ========== STEP 6: Save to Firebase ==========
        try:
            course_ref = get_course_ref(user_id, course_id)
            course_doc = course_ref.get()
            
            if course_doc.exists:
                # Update existing course - add this day to lessons_by_date
                course_data = course_doc.to_dict()
                lessons_by_date = course_data.get('lessons_by_date', {})
                lessons_by_date[day_date] = lesson_data
                
                course_ref.update({
                    'lessons_by_date': lessons_by_date
                })
                print(f"✅ Updated existing course with Day {day}")
            else:
                # Create new course document
                course_ref.set({
                    'joined_date': joined_date.strftime("%Y-%m-%d"),
                    'goal_name': goal_name,
                    'lessons_by_date': {
                        day_date: lesson_data
                    },
                    'created_at': datetime.now().isoformat()
                })
                print(f"✅ Created new course with Day {day}")
            
            print(f"✅ Saved to: users/{user_id}/datedcourses/{course_id}")
            
        except Exception as e:
            return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500
        
        # ========== STEP 7: Return Response ==========
        return jsonify({
            "success": True,
            "day": day,
            "date": day_date,
            "course_id": course_id,
            "lesson": lesson_data,
            "message": f"Day {day} lesson created successfully"
        })
    
    return final_plan_day_func

# ============ CREATE ALL ENDPOINTS ============
for i in range(1, 6):
    create_day_endpoint(i)

# ============ OPTIONAL: Batch Create All Days ==========
@app.route('/create-full-course', methods=['POST'])
def create_full_course():
    """Create all 5 days at once"""
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()
    user_answers = data.get("user_answers", [])
    user_id = data.get("user_id", "").strip()
    join_date_str = data.get("join_date")
    
    if not goal_name or not isinstance(user_answers, list) or not user_id:
        return jsonify({"error": "Missing required fields"}), 400
    
    results = []
    errors = []
    
    for day in range(1, 6):
        try:
            # Call each day endpoint internally
            endpoint_func = app.view_functions[f"final_plan_day_{day}"]
            # Note: This is simplified - in production, make actual HTTP calls
            results.append(f"Day {day} created")
        except Exception as e:
            errors.append(f"Day {day} failed: {str(e)}")
    
    return jsonify({
        "success": len(errors) == 0,
        "results": results,
        "errors": errors
    })

# ============ UTILITY: Get Course Progress ==========
@app.route('/get-course/<user_id>/<course_id>', methods=['GET'])
def get_course(user_id, course_id):
    """Get course data for debugging"""
    try:
        course_ref = get_course_ref(user_id, course_id)
        course_doc = course_ref.get()
        
        if not course_doc.exists:
            return jsonify({"error": "Course not found"}), 404
        
        return jsonify({
            "success": True,
            "data": course_doc.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



@app.route('/start-ai-helper', methods=['POST'])
def start_ai_helper():
    data = request.get_json()
    ai_plan = data.get("ai_plan")
    user_id = data.get("user_id")

    if not isinstance(ai_plan, dict):
        return jsonify({"error": "Missing or invalid ai_plan"}), 400

    prompt_template = load_prompt("prompt_ai_helper_start.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_ai_helper_start.txt not found"}), 500

    prompt = prompt_template.replace("<<ai_plan>>", json.dumps(ai_plan, indent=2))

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        save_to_firebase(user_id, "ai_helper_starts", {
            "ai_plan": ai_plan,
            "ai_intro": result
        })
        return jsonify({"ai_intro": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
@app.route('/ai-helper-reply', methods=['POST'])
def ai_helper_reply():
    data = request.get_json()
    ai_plan = data.get("ai_plan")
    chat_history = data.get("chat_history", [])
    user_id = data.get("user_id")

    if not isinstance(ai_plan, dict) or not isinstance(chat_history, list):
        return jsonify({"error": "Missing or invalid ai_plan or chat_history"}), 400

    history_text = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history if isinstance(m, dict)]
    )

    prompt_template = load_prompt("prompt_ai_helper_reply.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_ai_helper_reply.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<ai_plan>>", json.dumps(ai_plan, indent=2))
        .replace("<<chat_history>>", history_text)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()

        save_to_firebase(user_id, "ai_helper_replies", {
            "ai_plan": ai_plan,
            "chat_history": chat_history,
            "ai_reply": result
        })

        return jsonify({"ai_reply": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/daily-dashboard', methods=['POST'])
def daily_dashboard():
    data = request.get_json()
    day_number = data.get("day", 1)
    raw_html = data.get("goalplanner_saved_html", "")
    user_id = data.get("user_id")

    if not raw_html:
        return jsonify({"error": "Missing goalplanner_saved_html"}), 400

    soup = BeautifulSoup(raw_html, "html.parser")
    day_header = f"Skyler Day{day_number}"
    section = None

    for div in soup.find_all("div"):
        if day_header in div.text:
            section = div
            break

    if not section:
        return jsonify({"error": f"No content found for {day_header}"}), 404

    task_text = ""
    for p in section.find_all("p"):
        if p.find("strong") and "Task" in p.find("strong").text:
            task_text = p.text.replace("Task:", "").strip()
            break

    tasks = [t.strip() for t in task_text.split(",") if t.strip()]

    prompt_template = load_prompt("prompt_dashboard.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_dashboard.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<day>>", str(day_number))
        .replace("<<tasks>>", json.dumps(tasks, indent=2))
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)

        save_to_firebase(user_id, "dashboards", {
            "day": day_number,
            "tasks": tasks,
            "dashboard": parsed
        })

        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON from model", "raw_response": result}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/get-user-logs', methods=['GET'])
def get_all_logs():
    logs = read_logs()
    return jsonify({"logs": logs})

@app.route('/generate-reward-questions', methods=['POST'])
def generate_reward_questions():
    data = request.get_json()
    user_id = data.get("user_id", "")

    prompt_template = load_prompt("prompt_reward_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_reward_questions.txt not found"}), 500

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.5,
            max_tokens=400
        )
        questions = response.choices[0].message.content.strip()

        save_to_firebase(user_id, "reward_questions", {
            "questions": questions
        })

        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route('/analyze-reward', methods=['POST'])
def analyze_reward():
    data = request.get_json()
    user_id = data.get("user_id")
    answers = data.get("answers", [])

    if not user_id or not isinstance(answers, list) or len(answers) == 0:
        return jsonify({"error": "Missing user_id or answers"}), 400

    formatted_answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

    prompt_template = load_prompt("prompt_reward_analysis.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_reward_analysis.txt not found"}), 500

    prompt = prompt_template.replace("<<user_answers>>", formatted_answers)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=200
        )
        reward = response.choices[0].message.content.strip()

        rewards = read_rewards()
        rewards[user_id] = {
            "reward": reward,
            "task_completed": False
        }
        write_rewards(rewards)

        save_to_firebase(user_id, "rewards", {
            "answers": answers,
            "reward": reward
        })

        return jsonify({"reward": reward})
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route('/claim-reward', methods=['GET'])
def claim_reward():
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    rewards = read_rewards()
    if user_id not in rewards:
        return jsonify({"error": "No reward set for user"}), 404

    reward_data = rewards[user_id]

    return jsonify({"reward": reward_data.get("reward")})

@app.route('/complete-task', methods=['POST'])
def complete_task():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    rewards = read_rewards()
    if user_id not in rewards:
        return jsonify({"error": "User not found"}), 404

    rewards[user_id]["task_completed"] = True
    write_rewards(rewards)

    save_to_firebase(user_id, "task_completions", {
        "task_completed": True
    })

    return jsonify({"message": "Task marked complete. Reward unlocked!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)






































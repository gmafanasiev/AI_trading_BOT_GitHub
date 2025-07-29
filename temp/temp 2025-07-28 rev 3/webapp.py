from flask import Flask, render_template, request, jsonify
import os, subprocess, requests, time

app = Flask(__name__)
API_TOKEN = os.getenv('PA_API_TOKEN')  # Set this env var in dashboard or .env
USERNAME = 'gmafanasiev'  # Your username
HOST = 'www.pythonanywhere.com'  # Or 'eu.pythonanywhere.com' if EU server
BOT_DIR = '/home/gmafanasiev/ai_trading_bot'
CONSOLE_ID = None  # Global to track running console

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test_modules():
    try:
        result = subprocess.check_output(['python', os.path.join(BOT_DIR, 'test_heavy2.py')], stderr=subprocess.STDOUT, text=True, timeout=60)
        return jsonify({'output': result})
    except subprocess.CalledProcessError as e:
        return jsonify({'output': e.output.decode('utf-8') if isinstance(e.output, bytes) else e.output})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/start', methods=['POST'])
def start_bot():
    global CONSOLE_ID
    if CONSOLE_ID: return jsonify({'error': 'Bot already running'})
    # Create console via API (as bash)
    headers = {'Authorization': f'Token {API_TOKEN}'}
    payload = {'executable': '/bin/bash', 'arguments': '', 'working_directory': BOT_DIR}  # Start as bash
    print(f"Creating console: Headers: {headers}, Payload: {payload}")  # Debug
    resp = requests.post(f'https://{HOST}/api/v0/user/{USERNAME}/consoles/', headers=headers, json=payload)
    print(f"Create response: Status {resp.status_code}, Text: {resp.text}")
    if resp.status_code != 201:
        return jsonify({'error': resp.text})
    CONSOLE_ID = resp.json()['id']
    # Send command to run main.py
    send_payload = {'input': 'python main.py\n'}
    send_resp = requests.post(f'https://{HOST}/api/v0/user/{USERNAME}/consoles/{CONSOLE_ID}/send_input/', headers=headers, json=send_payload)
    print(f"Send input response: Status {send_resp.status_code}, Text: {send_resp.text}")
    if send_resp.status_code == 200:
        return jsonify({'message': 'Bot started', 'console_id': CONSOLE_ID})
    return jsonify({'error': send_resp.text})

@app.route('/stop', methods=['POST'])
def stop_bot():
    global CONSOLE_ID
    if not CONSOLE_ID: return jsonify({'error': 'No bot running'})
    headers = {'Authorization': f'Token {API_TOKEN}'}
    resp = requests.delete(f'https://{HOST}/api/v0/user/{USERNAME}/consoles/{CONSOLE_ID}/', headers=headers)
    if resp.status_code == 204:
        CONSOLE_ID = None
        return jsonify({'message': 'Bot stopped'})
    return jsonify({'error': resp.text})

@app.route('/monitor')
def monitor():
    try:
        with open(os.path.join(BOT_DIR, 'trade_alpaca_grok4.log'), 'r') as f:
            lines = f.readlines()[-500:]  # Last 500 lines
        return '<pre>' + ''.join(lines) + '</pre><script>setTimeout(() => location.reload(), 5000);</script>'  # Auto-refresh every 5s
    except Exception as e:
        return str(e)

@app.route('/params', methods=['GET', 'POST'])
def params():
    config_path = os.path.join(BOT_DIR, 'config.py')
    with open(config_path, 'r') as f:
        config_content = f.read()
    if request.method == 'POST':
        # Fetch form values with defaults
        new_symbols = request.form.get('SYMBOLS', '["AAPL"]')
        new_mode = request.form.get('TRADE_MODE', 'paper')
        new_sequence = request.form.get('SEQUENCE_LENGTH', '5')
        new_upper = request.form.get('UPPER_THRESHOLD', '0.65')
        new_lower = request.form.get('LOWER_THRESHOLD', '0.35')
        new_adjustment = request.form.get('ADJUSTMENT_INTERVAL', '30')
        new_max_position = request.form.get('MAX_POSITION_PCT', '0.10')
        new_max_equity = request.form.get('MAX_EQUITY', '100000.0')
        new_stop_loss = request.form.get('STOP_LOSS_PCT', '0.10')
        new_take_profit = request.form.get('TAKE_PROFIT_PCT', '0.06')
        new_trailing = request.form.get('TRAILING_PCT', '0.02')
        new_risk = request.form.get('RISK_PER_TRADE', '0.01')
        new_timeframe = request.form.get('TIMEFRAME', '"Minute"')
        new_cooldown = request.form.get('COOLDOWN_SECONDS', '60')
        new_session = request.form.get('SESSION_DURATION', '"10 minutes"')
        
        # Replace in config_content (assumes exact format in config.py)
        config_content = config_content.replace(f'SYMBOLS = {new_symbols}', f'SYMBOLS = {new_symbols}')
        config_content = config_content.replace(f"TRADE_MODE = '{new_mode}'", f"TRADE_MODE = '{new_mode}'")
        config_content = config_content.replace(f'SEQUENCE_LENGTH = {new_sequence}', f'SEQUENCE_LENGTH = {new_sequence}')
        config_content = config_content.replace(f'UPPER_THRESHOLD = {new_upper}', f'UPPER_THRESHOLD = {new_upper}')
        config_content = config_content.replace(f'LOWER_THRESHOLD = {new_lower}', f'LOWER_THRESHOLD = {new_lower}')
        config_content = config_content.replace(f'ADJUSTMENT_INTERVAL = {new_adjustment}', f'ADJUSTMENT_INTERVAL = {new_adjustment}')
        config_content = config_content.replace(f'MAX_POSITION_PCT = {new_max_position}', f'MAX_POSITION_PCT = {new_max_position}')
        config_content = config_content.replace(f'MAX_EQUITY = {new_max_equity}', f'MAX_EQUITY = {new_max_equity}')
        config_content = config_content.replace(f'STOP_LOSS_PCT = {new_stop_loss}', f'STOP_LOSS_PCT = {new_stop_loss}')
        config_content = config_content.replace(f'TAKE_PROFIT_PCT = {new_take_profit}', f'TAKE_PROFIT_PCT = {new_take_profit}')
        config_content = config_content.replace(f'TRAILING_PCT = {new_trailing}', f'TRAILING_PCT = {new_trailing}')
        config_content = config_content.replace(f'RISK_PER_TRADE = {new_risk}', f'RISK_PER_TRADE = {new_risk}')
        config_content = config_content.replace(f'TIMEFRAME = {new_timeframe}', f'TIMEFRAME = {new_timeframe}')
        config_content = config_content.replace(f'COOLDOWN_SECONDS = {new_cooldown}', f'COOLDOWN_SECONDS = {new_cooldown}')
        config_content = config_content.replace(f'SESSION_DURATION = {new_session}', f'SESSION_DURATION = {new_session}')
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        return 'Params updated'
    return render_template('params.html', config=config_content)

if __name__ == '__main__':
    app.run()
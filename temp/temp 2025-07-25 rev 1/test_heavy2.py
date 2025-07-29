# test_heavy2.py
import importlib
import inspect
import os
import logging
import warnings
import ast
import requests
import sys
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd  # Added for summary table
import json

warnings.filterwarnings('ignore', category=UserWarning, module='importlib', message='The NumPy module was reloaded')

for dep in ['numpy', 'pandas', 'sklearn', 'matplotlib']:
    if dep not in sys.modules:
        sys.modules[dep] = importlib.import_module(dep)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = '/home/gmafanasiev/ai_trading_bot'
HOME_DIR = '/home/gmafanasiev'

MODULES = {
    'config': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices'],
    'data_utils': ['fetch_bar_data', 'prepare_grok4_input'],
    'trade_analysis': ['analyze_trades'],
    'prediction': ['get_grok4_prediction_and_adjustments'],
    'trading_loop': ['trading_logic'],
    'order_execution': ['get_position', 'execute_order', 'poll_order_fill', 'update_position'],
    'GenerateAAPLJSON': ['AAPLDataProcessor']
}

DEPENDENCIES = ['pandas', 'numpy', 'sklearn', 'alpaca', 'aiohttp', 'dotenv', 'matplotlib']

CONFIG_IMPORTS = {
    'prediction': ['XAI_API_KEY', 'URL', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'TEMPERATURE', 'MARKET_CORR', 'SUPPORTS', 'RESISTANCES'],
    'trading_loop': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices', 'LONG_EXIT_THRESHOLD', 'SHORT_EXIT_THRESHOLD'],
    'order_execution': ['open_positions', 'sl_prices', 'tp_prices', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRADE_MODE', 'POLL_TIMEOUT'],
    'data_utils': ['TRADE_MODE', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'VOLATILITY']
}

# Collect results for summary table
results = []  # List of dicts: {'Category': str, 'Check': str, 'Status': str, 'Details': str}

def log_and_record(category, check, status, details=''):
    if status == 'OK':
        logger.info(f"{category:<25} | {check:<40}: {status:>10}")
    else:
        logger.error(f"{category:<25} | {check:<40}: {status:>10} - {details}")
    results.append({'Category': category, 'Check': check, 'Status': status, 'Details': details})

def print_section_header(header):
    logger.info("\n" + "=" * 80)
    logger.info(f"{header:^80}")
    logger.info("=" * 80 + "\n")

def check_dependencies():
    print_section_header("Dependencies Check")
    errors = []
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep)
            log_and_record('Dependencies', dep, 'OK')
        except ImportError as e:
            log_and_record('Dependencies', dep, 'Missing', f'Install with: pip install --user {dep}')
            errors.append(str(e))
    return errors

def check_env_file():
    print_section_header("Environment File Check")
    env_path = os.path.join(PROJECT_DIR, '.env')
    errors = []
    if not os.path.exists(env_path):
        log_and_record('Environment', '.env file', 'Missing', f'at {env_path}')
        errors.append(f"Missing .env file at {env_path}")
        return errors
    load_dotenv(env_path)
    required_keys = ['XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']
    for key in required_keys:
        if not os.getenv(key):
            log_and_record('Environment', f'.env key: {key}', 'Missing')
            errors.append(f"Missing {key} in .env")
    if not errors:
        log_and_record('Environment', '.env file', 'OK', 'All keys present')
    return errors

def check_xai_endpoint(config_module):
    print_section_header("xAI Endpoint Check")
    expected_url = "https://api.x.ai/v1/chat/completions"
    actual = getattr(config_module, 'URL', 'None')
    if actual == expected_url:
        log_and_record('xAI Endpoint', actual, 'OK')
        return True
    log_and_record('xAI Endpoint', actual, 'Incorrect', f'Expected: {expected_url}')
    return False

def check_model_prediction(prediction_module):
    print_section_header("Model Prediction Check")
    with open(prediction_module.__file__, 'r') as f:
        content = f.read()
    if '"model": "grok-3"' in content:
        log_and_record('Model', 'grok-3 in prediction.py', 'OK')
        return True
    log_and_record('Model', 'grok-3 in prediction.py', 'Missing/Incorrect')
    return False

def check_logging_config(logging_utils_module):
    print_section_header("Logging Configuration Check")
    try:
        logger, file_handler = logging_utils_module.setup_logging()
        handlers = logger.handlers
        has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in handlers)
        if has_file and has_stream:
            log_and_record('Logging', 'File and Stream handlers', 'OK')
            return True
        log_and_record('Logging', 'File and Stream handlers', 'Missing', 'FileHandler or StreamHandler')
        return False
    except Exception as e:
        log_and_record('Logging', 'Configuration', 'Error', str(e))
        return False

def check_cooldown_logic(trading_loop_module):
    print_section_header("Cooldown Logic Check")
    with open(trading_loop_module.__file__, 'r') as f:
        content = f.read()
    if "COOLDOWN_SECONDS" in content and "last_trade_time" in content:
        log_and_record('Cooldown', 'COOLDOWN_SECONDS and last_trade_time', 'OK')
        return True
    log_and_record('Cooldown', 'COOLDOWN_SECONDS and last_trade_time', 'Missing')
    return False

def check_trade_filtering(trade_analysis_module):
    print_section_header("Trade Filtering Check")
    with open(trade_analysis_module.__file__, 'r') as f:
        content = f.read()
    if "time_threshold" in content and "datetime" in content:
        log_and_record('Trade Filtering', 'time_threshold and datetime', 'OK')
        return True
    log_and_record('Trade Filtering', 'time_threshold and datetime', 'Missing')
    return False

def check_fetch_bar_data_signature(data_utils_module):
    print_section_header("Fetch Bar Data Signature Check")
    try:
        fetch_bar_data = getattr(data_utils_module, 'fetch_bar_data')
        sig = inspect.signature(fetch_bar_data)
        expected_params = ['data_client', 'symbol', 'timeframe', 'limit', 'file_handler']
        if list(sig.parameters.keys()) == expected_params:
            log_and_record('Signature', 'fetch_bar_data', 'OK')
            return True
        log_and_record('Signature', 'fetch_bar_data', 'Incorrect', f'Expected: {expected_params}, Found: {list(sig.parameters.keys())}')
        return False
    except Exception as e:
        log_and_record('Signature', 'fetch_bar_data', 'Error', str(e))
        return False

def check_api_connectivity(config_module):
    print_section_header("API Connectivity Check")
    try:
        headers = {"Authorization": f"Bearer {config_module.XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "grok-3", "messages": [{"role": "user", "content": "Test connectivity"}], "max_tokens": 10}
        response = requests.post(config_module.URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        log_and_record('Connectivity', 'xAI API', 'OK', 'Status 200')
        return True
    except Exception as e:
        log_and_record('Connectivity', 'xAI API', 'Failed', str(e))
        return False

def check_config_imports(module_name, module_path, expected_imports):
    print_section_header(f"Config Imports Check: {module_name}")
    errors = []
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=module_path)
        imported_vars = {name.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == 'config' for name in node.names}
        missing_imports = [var for var in expected_imports if var not in imported_vars]
        if missing_imports:
            log_and_record('Config Imports', module_name, 'Missing', ', '.join(missing_imports))
            errors.append(f"Missing config imports in {module_name}: {', '.join(missing_imports)}")
        else:
            log_and_record('Config Imports', module_name, 'OK')
    except Exception as e:
        log_and_record('Config Imports', module_name, 'Error', str(e))
        errors.append(f"Config import check error in {module_name}: {str(e)}")
    return errors

def check_parameters(config_module):
    print_section_header("Parameters Check")
    errors = []
    for param in ['UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'MAX_POSITION_PCT']:
        actual = getattr(config_module, param, None)
        if isinstance(actual, float) and 0 < actual < 1:
            log_and_record('Parameters', param, 'OK', f'Value: {actual}')
        else:
            log_and_record('Parameters', param, 'Invalid/Missing', f'Expected float (0-1), Found: {actual}')
            errors.append(f"Invalid {param}: Expected float (0-1), Found {actual}")
    for param in ['SEQUENCE_LENGTH', 'ADJUSTMENT_INTERVAL']:
        actual = getattr(config_module, param, None)
        if isinstance(actual, int) and actual > 0:
            log_and_record('Parameters', param, 'OK', f'Value: {actual}')
        else:
            log_and_record('Parameters', param, 'Invalid/Missing', f'Expected positive int, Found: {actual}')
            errors.append(f"Invalid {param}: Expected positive int, Found {actual}")
    for param in ['MAX_EQUITY']:
        actual = getattr(config_module, param, None)
        if isinstance(actual, float) and actual > 0:
            log_and_record('Parameters', param, 'OK', f'Value: {actual}')
        else:
            log_and_record('Parameters', param, 'Invalid/Missing', f'Expected positive float, Found: {actual}')
            errors.append(f"Invalid {param}: Expected positive float, Found {actual}")
    expected_modes = ['simulation', 'paper', 'live', 'real']
    trade_mode = getattr(config_module, 'TRADE_MODE', None)
    if trade_mode in expected_modes:
        log_and_record('Parameters', 'TRADE_MODE', 'OK', f'Selected: {trade_mode}')
    else:
        log_and_record('Parameters', 'TRADE_MODE', 'Invalid', f'Found: {trade_mode}')
        errors.append(f"Invalid TRADE_MODE: {trade_mode}")
    return errors

def check_module(module_name, expected_items):
    print_section_header(f"Module Check: {module_name}")
    module_path = os.path.join(HOME_DIR if module_name == 'GenerateAAPLJSON' else PROJECT_DIR, f"{module_name}.py")
    errors = []
    if not os.path.exists(module_path):
        if module_name == 'GenerateAAPLJSON':
            log_and_record('Modules', module_name, 'Optional', 'Not found, continuing')
            return errors
        log_and_record('Modules', module_name, 'Missing', f'at {module_path}')
        errors.append(f"Module {module_name} not found at {module_path}")
        return errors
    try:
        with open(module_path, 'r') as f:
            ast.parse(f.read())
        log_and_record('Modules', f'Syntax in {module_name}', 'OK')
    except SyntaxError as e:
        log_and_record('Modules', f'Syntax in {module_name}', 'Error', str(e))
        errors.append(f"Syntax error in {module_name}: {str(e)}")
        return errors
    if module_name in CONFIG_IMPORTS:
        errors.extend(check_config_imports(module_name, module_path, CONFIG_IMPORTS[module_name]))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        log_and_record('Modules', f'Import {module_name}', 'OK')
        missing_items = [item for item in expected_items if not hasattr(module, item)]
        for item in expected_items:
            if item in missing_items: continue
            obj = getattr(module, item)
            obj_type = 'Async function' if inspect.iscoroutinefunction(obj) else 'Sync function' if inspect.isfunction(obj) else type(obj).__name__
            log_and_record('Modules', f'Item: {item}', 'OK', obj_type)
        if missing_items:
            log_and_record('Modules', f'Items in {module_name}', 'Missing', ', '.join(missing_items))
            errors.append(f"Missing items in {module_name}: {', '.join(missing_items)}")
        else:
            log_and_record('Modules', f'All items in {module_name}', 'OK')
    except Exception as e:
        log_and_record('Modules', f'Import {module_name}', 'Error', str(e))
        errors.append(f"Import error in {module_name}: {str(e)}")
    return errors

def main():
    logger.info("Starting import consistency check for AI trading bot")
    all_errors = check_dependencies() + check_env_file()
    for module_name, items in MODULES.items():
        all_errors.extend(check_module(module_name, items))
    config_module = sys.modules.get('config')
    prediction_module = sys.modules.get('prediction')
    logging_utils_module = sys.modules.get('logging_utils')
    trading_loop_module = sys.modules.get('trading_loop')
    trade_analysis_module = sys.modules.get('trade_analysis')
    data_utils_module = sys.modules.get('data_utils')
    if config_module:
        all_errors.extend(check_parameters(config_module))
        if not check_xai_endpoint(config_module): all_errors.append("Incorrect xAI API endpoint")
        if not check_api_connectivity(config_module): all_errors.append("xAI API connectivity failed")
    if prediction_module and not check_model_prediction(prediction_module): all_errors.append("Incorrect model in prediction.py")
    if logging_utils_module and not check_logging_config(logging_utils_module): all_errors.append("Incorrect logging configuration")
    if trading_loop_module and not check_cooldown_logic(trading_loop_module): all_errors.append("Missing cooldown logic in trading_loop.py")
    if trade_analysis_module and not check_trade_filtering(trade_analysis_module): all_errors.append("Missing trade filtering in trade_analysis.py")
    if data_utils_module and not check_fetch_bar_data_signature(data_utils_module): all_errors.append("Incorrect fetch_bar_data signature in data_utils.py")
    print_section_header("Summary of All Checks")
    if results:
        df = pd.DataFrame(results)
        logger.info("\n" + df.to_string(index=False))
    if all_errors:
        logger.error(f"\nConsistency checks failed with {len(all_errors)} errors:")
        for i, error in enumerate(all_errors, 1):
            logger.error(f"  Error {i}: {error}")
        sys.exit(1)
    else:
        logger.info("\nAll checks passed! Ready to run main.py")

if __name__ == "__main__":
    lock_file = os.path.join(PROJECT_DIR, 'test_heavy2.lock')
    if os.path.exists(lock_file):
        logger.error(f"Script already running or did not exit cleanly: Remove {lock_file} and try again")
        sys.exit(1)
    try:
        with open(lock_file, 'w') as f:
            f.write('running')
        main()
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)
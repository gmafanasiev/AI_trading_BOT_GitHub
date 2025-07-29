# test_heavy4.py
import importlib
import inspect
import os
import logging
import ast
import requests
import sys
import argparse
from unittest.mock import Mock
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = '/home/gmafanasiev/ai_trading_bot'
HOME_DIR = '/home/gmafanasiev'

MODULES = {
    'config': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices', 'NUM_DAYS_HISTORY', 'BACKTEST_MODE'],
    'data_utils': ['fetch_bar_data', 'fetch_daily_data', 'prepare_grok4_input'],
    'trade_analysis': ['analyze_trades'],
    'prediction': ['get_grok4_prediction_and_adjustments', 'get_x_sentiment'],
    'trading_loop': ['trading_logic'],
    'order_execution': ['get_position', 'execute_order', 'poll_order_fill', 'update_position'],
    'backtest': ['backtest', 'MockTradingClient', 'MockAccount', 'MockClock'],
}

DEPENDENCIES = ['pandas', 'numpy', 'sklearn', 'alpaca', 'aiohttp', 'dotenv', 'matplotlib']

LIBRARY_ALIASES = {
    'np': 'numpy',
    'pd': 'pandas',
    'plt': 'matplotlib.pyplot',
}

CONFIG_IMPORTS = {
    'prediction': ['XAI_API_KEY', 'URL', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'TEMPERATURE', 'MARKET_CORR', 'SUPPORTS', 'RESISTANCES'],
    'trading_loop': ['SYMBOLS', 'SEQUENCE_LENGTH', 'UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'ADJUSTMENT_INTERVAL', 'MAX_POSITION_PCT', 'MAX_EQUITY', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'TIMEFRAME', 'URL', 'XAI_API_KEY', 'TRADE_MODE', 'open_positions', 'sl_prices', 'tp_prices', 'LONG_EXIT_THRESHOLD', 'SHORT_EXIT_THRESHOLD', 'RISK_PER_TRADE', 'NUM_DAYS_HISTORY'],
    'order_execution': ['open_positions', 'sl_prices', 'tp_prices', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRADE_MODE', 'POLL_TIMEOUT'],
    'data_utils': ['TRADE_MODE', 'ALPACA_API_KEY', 'ALPACA_API_SECRET', 'VOLATILITY', 'NUM_DAYS_HISTORY'],
    'backtest': ['ALPACA_API_KEY', 'ALPACA_API_SECRET', 'SYMBOLS', 'SEQUENCE_LENGTH', 'NUM_DAYS_HISTORY'],
}

# Results for summary table
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

def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_and_record(func.__name__.capitalize().replace('_', ' '), 'Execution', 'Error', str(e))
            return [str(e)]
    return wrapper

@error_handler
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

@error_handler
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

@error_handler
def check_xai_endpoint(config_module):
    print_section_header("xAI Endpoint Check")
    expected_url = "https://api.x.ai/v1/chat/completions"
    actual = getattr(config_module, 'URL', '')
    if actual == expected_url:
        log_and_record('xAI Endpoint', actual, 'OK')
        return True
    log_and_record('xAI Endpoint', actual, 'Incorrect', f'Expected: {expected_url}')
    return False

@error_handler
def check_model_prediction(prediction_module):
    print_section_header("Model Prediction Check")
    with open(prediction_module.__file__, 'r') as f:
        content = f.read()
    if '"model": "grok-3"' in content:
        log_and_record('Model', 'grok-3 in prediction.py', 'OK')
        return True
    log_and_record('Model', 'grok-3 in prediction.py', 'Missing/Incorrect')
    return False

@error_handler
def check_logging_config(logging_utils_module):
    print_section_header("Logging Configuration Check")
    logger, file_handler = logging_utils_module.setup_logging()
    handlers = logger.handlers
    has_file = any(isinstance(h, logging.FileHandler) for h in handlers)
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in handlers)
    if has_file and has_stream:
        log_and_record('Logging', 'File and Stream handlers', 'OK')
        return True
    log_and_record('Logging', 'File and Stream handlers', 'Missing', 'FileHandler or StreamHandler')
    return False

@error_handler
def check_cooldown_logic(trading_loop_module):
    print_section_header("Cooldown Logic Check")
    with open(trading_loop_module.__file__, 'r') as f:
        content = f.read()
    if "COOLDOWN_SECONDS" in content and "last_trade_time" in content:
        log_and_record('Cooldown', 'COOLDOWN_SECONDS and last_trade_time', 'OK')
        return True
    log_and_record('Cooldown', 'COOLDOWN_SECONDS and last_trade_time', 'Missing')
    return False

@error_handler
def check_trade_filtering(trade_analysis_module):
    print_section_header("Trade Filtering Check")
    with open(trade_analysis_module.__file__, 'r') as f:
        content = f.read()
    if "time_threshold" in content and "datetime" in content:
        log_and_record('Trade Filtering', 'time_threshold and datetime', 'OK')
        return True
    log_and_record('Trade Filtering', 'time_threshold and datetime', 'Missing')
    return False

@error_handler
def check_fetch_bar_data_signature(data_utils_module):
    print_section_header("Fetch Bar Data Signature Check")
    fetch_bar_data = getattr(data_utils_module, 'fetch_bar_data')
    sig = inspect.signature(fetch_bar_data)
    expected_params = ['data_client', 'symbol', 'timeframe', 'limit', 'file_handler']
    if list(sig.parameters.keys()) == expected_params:
        log_and_record('Signature', 'fetch_bar_data', 'OK')
        return True
    log_and_record('Signature', 'fetch_bar_data', 'Incorrect', f'Expected: {expected_params}, Found: {list(sig.parameters.keys())}')
    return False

@error_handler
def check_api_connectivity(config_module):
    print_section_header("API Connectivity Check")
    headers = {"Authorization": f"Bearer {config_module.XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": "Test connectivity"}], "max_tokens": 10}
    response = requests.post(config_module.URL, headers=headers, json=payload, timeout=5)
    response.raise_for_status()
    log_and_record('Connectivity', 'xAI API', 'OK', 'Status 200')
    return True

@error_handler
def check_config_imports(module_name, module_path, expected_imports):
    print_section_header(f"Config Imports Check: {module_name}")
    with open(module_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=module_path)
    imported_vars = {name.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == 'config' for name in node.names}
    missing_imports = [var for var in expected_imports if var not in imported_vars]
    if missing_imports:
        log_and_record('Config Imports', module_name, 'Missing', ', '.join(missing_imports))
        return [f"Missing config imports in {module_name}: {', '.join(missing_imports)}"]
    log_and_record('Config Imports', module_name, 'OK')
    return []

@error_handler
def check_module_dependencies(module_name, module_path):
    print_section_header(f"Module Dependencies Check: {module_name}")
    with open(module_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=module_path)
    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module
    errors = []
    for alias, lib in LIBRARY_ALIASES.items():
        if alias in used_names:
            if alias not in imports or imports[alias] != lib:
                log_and_record('Dependencies', f'{alias} in {module_name}', 'Missing', f'Expected: import {lib} as {alias}')
                errors.append(f"Missing import for {alias} (should be 'import {lib} as {alias}') in {module_name}")
            else:
                log_and_record('Dependencies', f'{alias} in {module_name}', 'OK')
        else:
            log_and_record('Dependencies', f'{alias} in {module_name}', 'Not Used')
    # Check for 'headers' in API calls (specific to prediction)
    if module_name == 'prediction':
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'post' and 'requests' in imports.values():
                headers_found = False
                for keyword in node.keywords:
                    if keyword.arg == 'headers':
                        headers_found = True
                        break
                if not headers_found:
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id == 'headers':
                            headers_found = True
                            break
                if not headers_found or 'headers' not in used_names:
                    log_and_record('Dependencies', f'headers in {module_name}', 'Missing', 'Variable headers used in API call but not defined')
                    errors.append(f"Variable 'headers' used in API call but not defined in {module_name}")
    return errors

@error_handler
def check_parameters(config_module):
    print_section_header("Parameters Check")
    errors = []
    float_params = ['UPPER_THRESHOLD', 'LOWER_THRESHOLD', 'STOP_LOSS_PCT', 'TAKE_PROFIT_PCT', 'TRAILING_PCT', 'MAX_POSITION_PCT']
    for param in float_params:
        actual = getattr(config_module, param, None)
        if isinstance(actual, float) and 0 < actual < 1:
            log_and_record('Parameters', param, 'OK', f'Value: {actual}')
        else:
            log_and_record('Parameters', param, 'Invalid/Missing', f'Expected float (0-1), Found: {actual}')
            errors.append(f"Invalid {param}: Expected float (0-1), Found {actual}")
    int_params = ['SEQUENCE_LENGTH', 'ADJUSTMENT_INTERVAL', 'NUM_DAYS_HISTORY']
    for param in int_params:
        actual = getattr(config_module, param, None)
        if isinstance(actual, int) and actual > 0:
            log_and_record('Parameters', param, 'OK', f'Value: {actual}')
        else:
            log_and_record('Parameters', param, 'Invalid/Missing', f'Expected positive int, Found: {actual}')
            errors.append(f"Invalid {param}: Expected positive int, Found {actual}")
    float_positive_params = ['MAX_EQUITY']
    for param in float_positive_params:
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

@error_handler
def check_module(module_name, expected_items):
    print_section_header(f"Module Check: {module_name}")
    module_path = os.path.join(PROJECT_DIR, f"{module_name}.py")
    errors = []
    if not os.path.exists(module_path):
        log_and_record('Modules', module_name, 'Missing', f'at {module_path}')
        errors.append(f"Module {module_name} not found at {module_path}")
        return errors
    with open(module_path, 'r') as f:
        ast.parse(f.read())
    log_and_record('Modules', f'Syntax in {module_name}', 'OK')
    errors.extend(check_module_dependencies(module_name, module_path))
    if module_name in CONFIG_IMPORTS:
        errors.extend(check_config_imports(module_name, module_path, CONFIG_IMPORTS[module_name]))
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
    return errors

@error_handler
def dynamic_execution_check(dynamic):
    if not dynamic:
        return []
    print_section_header("Dynamic Execution Check")
    errors = []
    # Mock dependencies for execution
    mock_trading_client = Mock()
    mock_data_client = Mock()
    mock_file_handler = Mock()
    mock_last_trade_time = {}
    current_price = 213.96
    # Test trading_logic (async)
    import trading_loop
    try:
        asyncio.run(trading_loop.trading_logic(mock_trading_client, mock_data_client, 'AAPL', current_price, 'log_file_path', mock_file_handler, mock_last_trade_time))
        log_and_record('Dynamic Execution', 'trading_logic', 'OK')
    except Exception as e:
        log_and_record('Dynamic Execution', 'trading_logic', 'Error', str(e))
        errors.append(str(e))
    # Add more dynamic tests for other functions if needed
    return errors

def main(dynamic=False):
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
        check_xai_endpoint(config_module)
        check_api_connectivity(config_module)
    if prediction_module:
        check_model_prediction(prediction_module)
    if logging_utils_module:
        check_logging_config(logging_utils_module)
    if trading_loop_module:
        check_cooldown_logic(trading_loop_module)
    if trade_analysis_module:
        check_trade_filtering(trade_analysis_module)
    if data_utils_module:
        check_fetch_bar_data_signature(data_utils_module)
    all_errors.extend(dynamic_execution_check(dynamic))
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
    parser = argparse.ArgumentParser(description='AI Trading Bot Consistency Checker')
    parser.add_argument('--dynamic', action='store_true', help='Run dynamic execution checks')
    args = parser.parse_args()
    lock_file = os.path.join(PROJECT_DIR, 'test_heavy4.lock')
    if os.path.exists(lock_file):
        logger.error(f"Script already running or did not exit cleanly: Remove {lock_file} and try again")
        sys.exit(1)
    try:
        with open(lock_file, 'w') as f:
            f.write('running')
        main(args.dynamic)
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)

import json

def LoadJSONConfiguration(path, configuration):
    with open(path, encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data[configuration]
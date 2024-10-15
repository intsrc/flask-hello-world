from flask import Flask, request, jsonify
from helpers.index import load_and_preprocess_icons, extract_icons_from_screenshot
import cv2
import numpy as np

app = Flask(__name__)

# Load icons once when the server starts
icons_dir = "icons"  # Update this path as needed
item_icons = load_and_preprocess_icons(icons_dir)

@app.route('/identify_items', methods=['POST'])
def identify_items():
    if 'screenshot' not in request.files:
        return jsonify({"error": "No screenshot provided"}), 400
    
    file = request.files['screenshot']
    screenshot_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    try:
        extracted_icons = extract_icons_from_screenshot(screenshot_array)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    identified_items = []
    for icon in extracted_icons:
        icon_gray = preprocess_image(icon)
        identified_item, match_score = identify_item(icon_gray, item_icons)
        identified_items.append({
            "item": identified_item,
            "match_score": float(match_score)
        })
    
    return jsonify({"identified_items": identified_items})
if __name__ == '__main__':
    app.run(debug=True)
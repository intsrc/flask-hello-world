from flask import Flask, request, jsonify, render_template_string
from helpers.index import load_and_preprocess_icons, extract_icons_from_screenshot
import cv2
import numpy as np
from vercel_blob import put, PutOptions, get

app = Flask(__name__)

# Load icons once when the server starts
icons_dir = "icons"  # Update this path as needed
item_icons = load_and_preprocess_icons(icons_dir)

@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Screenshot Upload and Identify</title>
        <style>
            body { font-family: Arial, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .upload-container { text-align: center; }
            #uploadBtn { padding: 10px 20px; font-size: 16px; cursor: pointer; }
            #results { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="upload-container">
            <input type="file" id="fileInput" style="display: none;" accept="image/*">
            <button id="uploadBtn">Upload Screenshot</button>
            <div id="results"></div>
        </div>
        <script>
            document.getElementById('uploadBtn').addEventListener('click', function() {
                document.getElementById('fileInput').click();
            });
            
            document.getElementById('fileInput').addEventListener('change', async function(event) {
                const file = event.target.files[0];
                if (file) {
                    try {
                        // Step 1: Get the presigned URL
                        const response = await fetch('/api/upload-url');
                        const { url, clientPayload } = await response.json();

                        // Step 2: Upload to Vercel Blob
                        await fetch(url, {
                            method: 'PUT',
                            body: file,
                            headers: {
                                'Content-Type': file.type,
                            },
                        });

                        // Step 3: Confirm the upload and identify items
                        const confirmResponse = await fetch('/api/confirm-upload', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(clientPayload),
                        });
                        const { blob, identified_items } = await confirmResponse.json();

                        // Display results
                        const resultsDiv = document.getElementById('results');
                        resultsDiv.innerHTML = '<h3>Identified Items:</h3>';
                        identified_items.forEach(item => {
                            resultsDiv.innerHTML += `<p>${item.item}: ${item.match_score.toFixed(2)}</p>`;
                        });
                    } catch (error) {
                        console.error('Error:', error);
                        alert('An error occurred while processing the screenshot.');
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/upload-url', methods=['GET'])
async def get_upload_url():
    options = PutOptions(access='public')
    client_payload, url = await put('screenshots/image.png', options)
    return jsonify({"url": url, "clientPayload": client_payload})

@app.route('/api/confirm-upload', methods=['POST'])
async def confirm_upload():
    client_payload = request.json
    blob = await put('screenshots/image.png', client_payload)
    
    # Download the image from Vercel Blob
    image_data = await get(blob.url)
    nparr = np.frombuffer(image_data, np.uint8)
    screenshot_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image
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
    
    return jsonify({"blob": blob, "identified_items": identified_items})

if __name__ == '__main__':
    app.run(debug=True)
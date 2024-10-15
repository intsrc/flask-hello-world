import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_image(image, resize_dim=None, apply_blur=True, apply_threshold=False):
    """
    Preprocesses the image by resizing, converting to grayscale, blurring, and thresholding.

    Parameters:
        image (numpy.ndarray): The original BGR image.
        resize_dim (tuple or None): New size as (width, height). If None, no resizing.
        apply_blur (bool): Whether to apply Gaussian Blur.
        apply_threshold (bool): Whether to apply adaptive thresholding.

    Returns:
        numpy.ndarray: The preprocessed grayscale image.
    """
    if resize_dim:
        image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if apply_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if apply_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    cv2.imwrite('preprocessed_image.png', gray)
    return gray

def load_and_preprocess_icons(icons_directory, resize_dim=None, apply_blur=True, apply_threshold=False):
    """
    Loads and preprocesses all icon images from the specified directory.

    Parameters:
        icons_directory (str): Path to the directory containing icon images.
        resize_dim (tuple or None): Resize dimensions for icons.
        apply_blur (bool): Whether to apply Gaussian Blur.
        apply_threshold (bool): Whether to apply adaptive thresholding.

    Returns:
        dict: A dictionary with item names as keys and preprocessed icon images as values.
    """
    preprocessed_icons = {}
    for icon_filename in os.listdir(icons_directory):
        icon_path = os.path.join(icons_directory, icon_filename)
        icon_image = cv2.imread(icon_path, cv2.IMREAD_COLOR)
        if icon_image is None:
            print(f"Warning: Unable to load icon image at path: {icon_path}. Skipping.")
            continue
        
        preprocessed_icon = preprocess_image(
            icon_image,
            resize_dim=resize_dim,
            apply_blur=apply_blur,
            apply_threshold=apply_threshold
        )
        
        item_name = os.path.splitext(icon_filename)[0]
        preprocessed_icons[item_name] = preprocessed_icon
    
    return preprocessed_icons

def identify_item(slot_gray, item_icons, threshold=0.4):
    """
    Identifies the item in the slot image by matching it against preprocessed item icons.

    Parameters:
        slot_gray (numpy.ndarray): Preprocessed grayscale slot image.
        item_icons (dict): Dictionary of preprocessed icon images.
        threshold (float): Matching threshold between 0 and 1.

    Returns:
        tuple: Identified item name and its match value. Returns (None, 0) if no match is found.
    """
    max_val = 0
    identified_item = None

    # Define a helper function for parallel matching
    def match_icon(item_name, icon_gray):
        result = cv2.matchTemplate(slot_gray, icon_gray, cv2.TM_CCOEFF_NORMED)
        _, current_max_val, _, _ = cv2.minMaxLoc(result)
        return (item_name, current_max_val)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        future_to_item = {executor.submit(match_icon, name, icon): name for name, icon in item_icons.items()}
        for future in as_completed(future_to_item):
            item_name, current_max_val = future.result()
            # Debug: Print match value
            print(f"Matching {item_name}: Match Value = {current_max_val:.4f}")
            if current_max_val > max_val and current_max_val >= threshold:
                max_val = current_max_val
                identified_item = item_name

    return identified_item, max_val

def draw_rectangles(slot_image, slot_gray, icon_gray, threshold=0.4):
    """
    Draws rectangles around matched regions in the slot image.

    Parameters:
        slot_image (numpy.ndarray): Original BGR slot image.
        slot_gray (numpy.ndarray): Preprocessed grayscale slot image.
        icon_gray (numpy.ndarray): Preprocessed grayscale icon image.
        threshold (float): Matching threshold.

    Returns:
        numpy.ndarray: The slot image with rectangles drawn.
    """
    res = cv2.matchTemplate(slot_gray, icon_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = icon_gray.shape

    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(slot_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    
    return slot_image

def extract_icons_from_screenshot(screenshot):
    """
    Extracts 9 icons from a Dota 2 screenshot based on the specified layout,
    supporting both 1080p and 2K resolutions.

    Parameters:
        screenshot_path (str): Path to the Dota 2 screenshot.

    Returns:
        list: A list of 6 numpy arrays, each representing an extracted icon.
    """
    if screenshot is None:
        raise ValueError(f"Unable to load screenshot from path: {screenshot_path}")

    height, width = screenshot.shape[:2]

    if height == 1080:  # 1080p resolution
        start_x = 1151
        start_y = height - 137
        icon_width, icon_height = 60, 45
        horizontal_gap = 6
        vertical_gap = 3
    elif height == 1620:  # 2K resolution
        start_x = 1703
        start_y = height - 205
        icon_width, icon_height = 90, 68
        horizontal_gap = 8
        vertical_gap = 4
    else:
        raise ValueError(f"Unsupported resolution: {width}x{height}")

    icons = []
    for row in range(2):
        for col in range(3):
            x = start_x + col * (icon_width + horizontal_gap)
            y = start_y + row * (icon_height + vertical_gap)
            icon = screenshot[y:y+icon_height, x:x+icon_width]
            icons.append(icon)

    return icons

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Identify items in a Dota 2 screenshot using template matching.")
    parser.add_argument('--screenshot', type=str, required=True, help='Path to the Dota 2 screenshot.')
    parser.add_argument('--icons_dir', type=str, required=True, help='Directory containing icon images.')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output images.')
    parser.add_argument('--threshold', type=float, default=0.4, help='Matching threshold between 0 and 1.')
    parser.add_argument('--resize_icon', type=int, nargs=2, metavar=('width', 'height'), help='Resize icon images to specified width and height.')
    parser.add_argument('--blur', action='store_true', help='Apply Gaussian Blur to images.')
    parser.add_argument('--thresholding', action='store_true', help='Apply adaptive thresholding to images.')
    
    args = parser.parse_args()

    # Extract icons from the screenshot
    extracted_icons = extract_icons_from_screenshot(args.screenshot)
    print(f"Extracted {len(extracted_icons)} icons from the screenshot.")

    # Load and preprocess icon images from the icons directory
    item_icons = load_and_preprocess_icons(
        args.icons_dir,
        resize_dim=tuple(args.resize_icon) if args.resize_icon else None,
        apply_blur=args.blur,
        apply_threshold=args.thresholding
    )

    if not item_icons:
        print("Error: No valid icon images found in the icons directory.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # Process each extracted icon
    for i, extracted_icon in enumerate(extracted_icons):
        print(f"\nProcessing extracted icon {i+1}:")
        
        # Preprocess the extracted icon
        extracted_icon_gray = preprocess_image(
            extracted_icon,
            resize_dim=tuple(args.resize_icon) if args.resize_icon else None,
            apply_blur=args.blur,
            apply_threshold=args.thresholding
        )

        # Identify the item
        identified_item, match_score = identify_item(extracted_icon_gray, item_icons, threshold=args.threshold)

        # Store the results instead of printing them immediately
        if identified_item:
            results.append(f"Icon {i+1}: Identified Item: {identified_item} with a match score of {match_score:.4f}")
            
            # Draw rectangles on the extracted icon
            icon_gray = item_icons[identified_item]
            icon_with_rectangles = draw_rectangles(extracted_icon.copy(), extracted_icon_gray, icon_gray, threshold=args.threshold)
            
            # Save the output image
            output_path = os.path.join(args.output_dir, f"identified_icon_{i+1}.png")
            cv2.imwrite(output_path, icon_with_rectangles)
            print(f"Output image with matches saved to {output_path}")
        else:
            results.append(f"Icon {i+1}: No matching item found")

    # Print all results after processing is complete
    print("\n--- Final Results ---")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
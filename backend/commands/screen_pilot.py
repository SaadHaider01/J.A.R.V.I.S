import cv2
import numpy as np
import pyautogui
import logging
import os
import time

logger = logging.getLogger("JARVIS.ScreenPilot")

# Base folder for UI templates
TEMPLATES_FOLDER = os.path.join(os.getcwd(), "frontend", "assets", "templates")
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)

def _get_template_path(image_name: str) -> str:
    """Resolves the image template to either an absolute path or relative to frontend assets."""
    if not image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_name += ".png"
        
    path = os.path.join(TEMPLATES_FOLDER, image_name)
    if os.path.exists(image_name):
        return image_name
    return path

def _locate_template(image_path: str, threshold: float = 0.8):
    """
    Takes a screenshot, uses OpenCV template matching, and returns the (x, y) 
    center coordinates of the match or None.
    """
    from backend.commands.cancel_handler import check_cancelled

    final_path = _get_template_path(image_path)
    if not os.path.exists(final_path):
        logger.error(f"Template not found: {final_path}")
        return None

    logger.info(f"Taking screenshot to find template: {final_path}")
    screenshot = pyautogui.screenshot()
    img_rgb = np.array(screenshot)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.error("Failed to load template image.")
        return None
        
    w, h = template.shape[::-1]

    # Perform OpenCV match operations
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if check_cancelled(): return None

    if max_val >= threshold:
        # returns central point
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return (center_x, center_y)
        
    logger.warning(f"Template '{image_path}' not found! Confidence: {max_val:.2f} < {threshold}")
    return None

def find_and_click(image_path: str) -> bool:
    """Takes a screenshot, finds a UI element visually, and clicks it."""
    coords = _locate_template(image_path)
    if coords:
        try:
            logger.info(f"Clicking matched element at {coords}")
            pyautogui.moveTo(coords[0], coords[1], duration=0.2)
            pyautogui.click()
            return True
        except Exception as e:
            logger.error(f"find_and_click failed: {e}")
    return False

def find_and_type(image_path: str, text: str) -> bool:
    """Finds an input field visually and types into it."""
    if find_and_click(image_path):
        time.sleep(0.5)
        pyautogui.write(text)
        return True
    return False

def scroll_to_element(image_path: str, max_scrolls: int = 15) -> bool:
    """Scrolls until a visual element appears on screen."""
    from backend.commands.cancel_handler import check_cancelled
    
    for _ in range(max_scrolls):
        if check_cancelled(): return False
        
        coords = _locate_template(image_path)
        if coords:
            return True
            
        logger.info("Element not found. Scrolling down...")
        pyautogui.scroll(-300)
        time.sleep(0.5)
        
    return False

def double_click_element(image_path: str) -> bool:
    """Double clicks any visible element."""
    coords = _locate_template(image_path)
    if coords:
        pyautogui.moveTo(coords[0], coords[1], duration=0.2)
        pyautogui.doubleClick()
        return True
    return False

def right_click_element(image_path: str) -> bool:
    """Right clicks for context menus."""
    coords = _locate_template(image_path)
    if coords:
        pyautogui.moveTo(coords[0], coords[1], duration=0.2)
        pyautogui.rightClick()
        return True
    return False

def drag_element(source_image: str, target_image: str) -> bool:
    """Drags one element to another visually."""
    source_coords = _locate_template(source_image)
    if not source_coords: return False
    
    target_coords = _locate_template(target_image)
    if not target_coords: return False
    
    logger.info(f"Dragging from {source_coords} to {target_coords}")
    pyautogui.moveTo(source_coords[0], source_coords[1], duration=0.2)
    pyautogui.dragTo(target_coords[0], target_coords[1], duration=0.5, button='left')
    return True

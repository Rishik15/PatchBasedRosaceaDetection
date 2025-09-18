import cv2
import os

IMG_DIR = "data/train/rosacea/"
OUT_FILE = "data/train/rosacea/coord.dat"

clicks = []
current_filename = ""

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Clicked at ({x}, {y})")

def processedImages(out_file):
    processed = set()
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    processed.add(parts[0])
    return processed

def annotate_images():
    global clicks, current_filename

    processed = processedImages(OUT_FILE)

    with open(OUT_FILE, 'a') as out_f:
        for filename in sorted(os.listdir(IMG_DIR)):
            if not filename.lower().endswith(('.jpg', '.png', '.pgm')):
                continue
            if filename in processed:
                continue
            # if filename.lower() != '4.png':
            #     continue

            img_path = os.path.join(IMG_DIR, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {filename}")
                continue

            current_filename = filename
            clicks = []

            print(f"Annotating {filename}. Click on LEFT eye, then RIGHT eye.")
            cv2.imshow("Image", img)
            cv2.setMouseCallback("Image", mouse_callback)

            # Wait until 2 points are clicked
            while len(clicks) < 2:
                cv2.waitKey(1)

            # Sort by x-coordinate to ensure left-right order
            (x1, y1), (x2, y2) = sorted(clicks, key=lambda pt: pt[0])
            out_f.write(f"{filename} {x1} {y1} {x2} {y2}\n")
            print(f"Saved: {filename} {x1} {y1} {x2} {y2}")
            cv2.destroyAllWindows()

annotate_images()
from PIL import Image, ImageDraw
import numpy as np

w = 130

h = 150

c = 3

new_seed = 42

np.random.seed(new_seed)

patch_configs = {
    "small": {
        "forehead_point": (48, 20),
        "nose_point": (55, 75),
        "left_cheek_point": (12, 77),
        "right_cheek_point": (105, 85),
        "forehead_triangle": {"height": 8, "base": 25},
        "nose_rect": (15, 15),
        "left_rect": (15, 15),
        "right_rect": (10, 15),
    },
    "medium": {
        "forehead_point": (27, 10),
        "nose_point": (50, 65),
        "left_cheek_point": (9, 73),
        "right_cheek_point": (92, 76),
        "forehead_triangle": {"height": 17, "base": 70},
        "nose_rect": (28, 26),
        "left_rect": (27, 37),
        "right_rect": (27, 33),
    },
    "large": {
        "forehead_point": (10, 3),
        "nose_point": (48, 54),
        "left_cheek_point": (9, 64),
        "right_cheek_point": (87, 66),
        "forehead_triangle": {"height": 30, "base": 110},
        "nose_rect": (33, 40),
        "left_rect": (35, 45),
        "right_rect": (37, 45),
    }
}

def maskImages(config, forehead, nose, left, right):
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if forehead:
        tri = config["forehead_triangle"]
        p = config["forehead_point"]
        triangle_points = [
            (p[0], p[1]),
            (p[0] + tri["base"], p[1]),
            (p[0] + tri["base"] // 2, p[1] + tri["height"])
        ]
        draw.polygon(triangle_points, fill=255)

    if nose:
        p = config["nose_point"]
        w_n, h_n = config["nose_rect"]
        draw.rectangle([p, (p[0] + w_n, p[1] + h_n)], fill=255)

    if left:
        p = config["left_cheek_point"]
        w_l, h_l = config["left_rect"]
        draw.rectangle([p, (p[0] + w_l, p[1] + h_l)], fill=255)

    if right:
        p = config["right_cheek_point"]
        w_r, h_r = config["right_rect"]
        draw.rectangle([p, (p[0] + w_r, p[1] + h_r)], fill=255)

    mask_array = np.array(mask, dtype=np.float32) / 255.0
    mask = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)
    return mask
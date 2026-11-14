import os
import random
from PIL import Image, ImageDraw

random.seed(0)

def make(cls, path):
    bg = random.randint(0, 80)

    img = Image.new("RGB", (150, 150), (bg, bg, bg))
    d = ImageDraw.Draw(img)

    cx, cy = random.randint(55, 95), random.randint(55, 95)
    r = random.randint(20, 40)

    color = (
        random.randint(150, 255),
        random.randint(0, 255),
        random.randint(0, 100)
    )

    box = [cx - r, cy - r, cx + r, cy + r]

    if cls == "a":
        d.ellipse(box, fill=color)
    else:
        d.rectangle(box, fill=color)

    img.save(path)


for split, n in (("train", 200), ("test", 50)):
    for cls in ("a", "b"):
        os.makedirs(f"{split}/{cls}", exist_ok=True)

        for i in range(n):
            make(cls, f"{split}/{cls}/img_{i:03d}.png")

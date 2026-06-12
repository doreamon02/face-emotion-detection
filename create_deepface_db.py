import os
import cv2
import pickle
from tqdm import tqdm
from deepface import DeepFace

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = r"D:\23BTRCL054_sorted"
OUTPUT_DB = r"D:\23BTRCL054_deepface_db.pkl"
FAILED_LOG = r"D:\failed_images.txt"

# Allowed extensions
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ------------------------------
# CLEAN FILENAME FUNCTION
# ------------------------------
def clean_filename(path):
    folder, filename = os.path.split(path)
    new_name = filename.replace(",", "").replace(" ", "").replace("(", "").replace(")", "")
    new_path = os.path.join(folder, new_name)

    if new_path != path:
        os.rename(path, new_path)

    return new_path

# ------------------------------
# FACE EMBEDDING EXTRACTION WITH MULTIPLE DETECTORS
# ------------------------------
def extract_embedding(img_path):
    detectors = ["opencv", "ssd", "mtcnn", "retinaface"]
    for d in detectors:
        try:
            embedding = DeepFace.represent(
                img_path=img_path, 
                model_name="ArcFace", 
                detector_backend=d,
                enforce_detection=False
            )
            if embedding and len(embedding) > 0:
                return embedding[0]["embedding"]
        except:
            continue
    return None

# ------------------------------
# MAIN PROCESSING
# ------------------------------

people = sorted([p for p in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, p))])
print(f"Found {len(people)} people: {people}\n")

database = {}
failed = []

for person in people:
    person_dir = os.path.join(INPUT_DIR, person)
    images = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in VALID_EXT]

    print(f"\nProcessing {person} - {len(images)} images")
    embeddings = []

    for img_name in tqdm(images):
        img_path = os.path.join(person_dir, img_name)

        # Clean filename
        img_path = clean_filename(img_path)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            failed.append((img_path, "Unreadable image"))
            continue

        # Extract face embedding
        emb = extract_embedding(img_path)
        if emb is None:
            failed.append((img_path, "Face not detected"))
            continue

        embeddings.append((img_path, emb))

    database[person] = embeddings

# ------------------------------
# SAVE DB
# ------------------------------
with open(OUTPUT_DB, "wb") as f:
    pickle.dump(database, f)

print(f"\n✅ Saved DB to: {OUTPUT_DB}")

# ------------------------------
# SAVE FAILED IMAGES LOG
# ------------------------------
with open(FAILED_LOG, "w") as f:
    for item in failed:
        f.write(f"{item[0]} --> {item[1]}\n")

print(f"⚠️ Failed images: {len(failed)} (full list saved to {FAILED_LOG})")
print("Done ✔")


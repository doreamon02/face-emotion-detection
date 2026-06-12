import os
from deepface import DeepFace
import cv2
import csv

# CHANGE THIS to your dataset folder (same you used for DB)
DATASET_DIR = r"D:\23BTRCL054_sorted"

OUTPUT_CSV = r"D:\emotion_results.csv"
FAILED_FILE = r"D:\emotion_failed.txt"

def analyze_emotions():
    emotion_data = []
    failed_images = []

    # Walk through all subfolders
    for person_folder in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_folder)

        if not os.path.isdir(person_path):
            continue

        print(f"\n🔍 Processing folder: {person_folder}\n")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            # skip non images
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            print(f"  → Analyzing: {img_name}")

            try:
                result = DeepFace.analyze(
                    img_path,
                    actions=['emotion'],
                    enforce_detection=False
                )

                emotion = result['dominant_emotion']

                emotion_data.append([person_folder, img_name, emotion])

            except Exception as e:
                print(f"    ⚠️ Failed: {img_name}")
                failed_images.append(img_path)

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Person", "Filename", "Emotion"])
        writer.writerows(emotion_data)

    # Save failed list
    with open(FAILED_FILE, "w") as f:
        for item in failed_images:
            f.write(item + "\n")

    print("\n==============================")
    print(f"✅ Emotion detection completed!")
    print(f"📄 Saved results: {OUTPUT_CSV}")
    print(f"⚠️ Failed images: {len(failed_images)}")
    print(f"📁 Failed list saved: {FAILED_FILE}")
    print("==============================")

if __name__ == "__main__":
    analyze_emotions()

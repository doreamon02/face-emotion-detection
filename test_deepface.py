from deepface import DeepFace
import tensorflow as tf

print("TF:", tf.__version__)
DeepFace.build_model("Facenet")

print("DeepFace is working correctly.")

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

embedding_file_path = "embeddings/face_embeddings.pickle"
svc_model_file = "svm_model/svm_model.pickle"
label_encoder_file = "svm_model/label_encoding.pickle"

print("reading Embedding file...")
with open(embedding_file_path, "rb") as file:
    file_reader = file.read()
    embeddings = pickle.loads(file_reader)
print("Embedding loaded.")


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(embeddings["names"])
print("Label Encoder initialised and labels encoded")

#  print(embeddings["names"]) ['keerthivasan', 'keerthivasan'....]
#  print(labels) [0, 0, 0, 0, 0....]

print("Initialising Support Vector Machine...")
svc_classifier = SVC(C=1.0, kernel="linear", probability=True)
svc_classifier.fit(embeddings["embeddings"], labels)
print("SVC training completed")

with open(svc_model_file, 'wb') as file:
    file.write(pickle.dumps(svc_classifier))

with open(label_encoder_file, 'wb') as file:
    file.write(pickle.dumps(label_encoder))



import os
import face_recognition
filePath="image/"
filetype=".jpg"
fileList = []
for top, dirs, nondirs in os.walk(filePath):
    for item in nondirs:
        if filetype in item:
            fileList.append(os.path.join(top, item))
# print(fileList)
# Load a sample picture and learn how to recognize it.
known_face_encodings = []
known_face_names = []
i=0
for path in fileList:
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(path))[0].tolist())
    known_face_names.append(path.rstrip(".jpg").lstrip("image/"))
    i=i+1
file=open("data/face.txt",'w')
file.write(str(known_face_encodings))
file.close()
file=open("data/name.txt",'w')
file.write(str(known_face_names))
file.close()

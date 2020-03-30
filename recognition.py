# module and library required to build a Face Recognition System
import face_recognition
import datetime
import time
import cv2

# objective: this code willl help you in running face recognition on a video file and saving the results to a new video file.
# Open the input movie file
# "VideoCapture" is a class for video capturing from video files, image sequences or cameras

input_video = cv2.VideoCapture("future.mp4")

# "CAP_PROP_FRAME_COUNT": it helps in finding number of frames in the video file.

length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file(make sure resolution/frame rate matches input video!)
# So we capture a video, process it frame-by-frame and we want to save that video, it only possible by using "VideoWriter" object
# FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in courcc.org. It is platform dependent.

fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')

#25.07- number of frames per second(fps)
#(1280, 720)- frame size

output_video = cv2.VideoWriter('output.avi', fourcc, 25.07, (1280, 620))

# Load some sample pictures and learn how to recognize them
female_image = face_recognition.load_image_file("3.png")
female_face_encoding = face_recognition.face_encodings(female_image)[0]

# "face_recognition.face_encodings": it's a face_recognition package which returns a list of 128-dimensional face encodings

male_image = face_recognition.load_image_file("2.png")
male_face_encoding = face_recognition.face_encodings(male_image)[0]

known_faces = [
    female_face_encoding,
    male_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_video.read()
    frame = cv2.resize(frame, (640, 360))
    frame_number += 1
    
    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color(which OpenCV uses) to RGB color(which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    before = datetime.datetime.now()
    face_locations = face_recognition.face_locations(rgb_frame)
    after = datetime.datetime.now()
    print(f"Detection time: {(after - before).total_seconds()}")
    before = datetime.datetime.now()
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    after = datetime.datetime.now()
    print(f"Encoding time: {(after - before).total_seconds()}")

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.30)
        
        name = None
        if match[0]:
            name = "Jon.YJ"
        elif match[1]:
            name = "Aayush"
        face_names.append(name)
    
    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom-25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom -6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Result", cv2.resize(frame, (640, 360)))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # time.sleep(1)
    # Write the resultinng image to the output video file
    # print("Writing frame {} / {}".format(frame_number, length))
    # output_video.write(frame)

# All done!
input_video.release()
cv2.destroyAllWindows()

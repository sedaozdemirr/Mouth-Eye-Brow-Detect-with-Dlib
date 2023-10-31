import dlib
import cv2
import numpy as np
from skimage import color

# Load the input image
img ="faceset/5.jpg.jpg" #
def skinDetect(img):
    image = cv2.imread(img)
    # Create a HOG face detector
    face_detector = dlib.get_frontal_face_detector()
    
    # Detect faces in the image
    faces = face_detector(image, 1)
    
    # Loop over the detected faces
    for face in faces:
        # Get the landmarks for the face
        landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")(image, face)
    
        # Extract the mouth from the image
        mouth_points = []
        l_eyebrow = []
        r_eyebrow = []
        l_eye = []
        r_eye = []

        for i in range(17,22): #left brow
            l_eyebrow.append((landmarks.part(i).x, landmarks.part(i).y))
        l_eyebrow = np.array(l_eyebrow, np.int32)
        cv2.polylines(image, [l_eyebrow], True, (255, 0, 0), 2)
        cv2.fillPoly(image, [l_eyebrow], (255, 255, 255))
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [l_eyebrow], (255, 255, 255))
        mask = cv2.bitwise_not(mask)
        lfteyeb_image = cv2.bitwise_and(image, mask)
    
        for i in range(22,27): #rigth brow
            r_eyebrow.append((landmarks.part(i).x, landmarks.part(i).y))
        r_eyebrow = np.array(r_eyebrow, np.int32)
        cv2.polylines(image, [r_eyebrow], True, (255, 0, 0), 2)
        cv2.fillPoly(image, [r_eyebrow], (0, 0, 0))
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [r_eyebrow], (255, 255, 255))
        mask = cv2.bitwise_not(mask)
        rghteyeb_image = cv2.bitwise_and(lfteyeb_image, mask)
    
        for i in range(36,42): #left eye
            l_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        l_eye = np.array(l_eye, np.int32)
        cv2.polylines(image, [l_eye], True, (255, 0, 0), 2)
        cv2.fillPoly(image, [l_eye], (255, 255, 255))
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [l_eye], (255, 255, 255))
        mask = cv2.bitwise_not(mask)
        leye_image = cv2.bitwise_and(rghteyeb_image, mask)
    
        for i in range(42,48): #rigth eye
            r_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        r_eye = np.array(r_eye, np.int32)
        cv2.polylines(image, [r_eye], True, (255, 0, 0), 2)
        cv2.fillPoly(image, [r_eye], (255, 255, 255))
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [r_eye], (255, 255, 255))
        mask = cv2.bitwise_not(mask)
        seye_image = cv2.bitwise_and(leye_image, mask)
    
        
        for i in range(48, 61):
            mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
        mouth_points = np.array(mouth_points, np.int32)
        cv2.polylines(image, [mouth_points], True, (255, 0, 0), 2)
        cv2.fillPoly(image, [mouth_points], (255, 255, 255))
        mask = np.zeros(image.shape, np.uint8)
        cv2.fillPoly(mask, [mouth_points], (255, 255, 255))
        mask = cv2.bitwise_not(mask)
        mouth_image = cv2.bitwise_and(seye_image, mask)
    
        try:
            cv2.imwrite(f'dlibset/1.jpg',mouth_image) #output data files path
        except:
            pass
        
    # Save the output image
    rgb_image = np.array(mouth_image, dtype=np.uint8)  # red image
    lab_image = color.rgb2lab(rgb_image)
    print(lab_image)
    cv2.imwrite("output.jpg", mouth_image)
    cv2.imshow("output", mouth_image)
    cv2.waitKey(0)
    
    # Tüm pencereleri kapatma
    cv2.destroyAllWindows()

skinDetect(img)
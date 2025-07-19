import mediapipe as mp 
import math
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 


from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
mp.tasks
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """one image with Prediced category name and scores"""
    print("HERE IS SUBPLOT",subplot)
    # if subplot is not None:
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)




def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):

    """ Display batches"""
    images=[  image.numpy_view() for image in images]
    print("HERE IS I..........IMAGE",len(images))
    gestures= [top_gesture for (top_gesture,_) in results]
    multi_hand_landmarks_list=[multi_hand_landmarks for (_, multi_hand_landmarks) in results]
    

    rows = int(math.sqrt(len(images)))
    print("HERE IS ROWS",rows)
    cols = len(images) // rows
    print("HERE IS COLS",cols)
    
    print("rows", rows)
    print("cols",cols)
    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()






#  create an GestureRecognizer Object. 
model_path='models/gesture_recognizer.task'
base_options=python.BaseOptions(model_asset_path=model_path)
options=vision.GestureRecognizerOptions(base_options=base_options)
recognizer=vision.GestureRecognizer.create_from_options(options)

Image_common_path='/images'
IMAGE_FILENAMES=[
                'images/thumbs_down.jpg',
                'images/victory.jpg',
                'images/thumbs_up.jpg',
                'images/pointing_up.jpg' 
                ]
images=[]
results=[]
count=0
for image_file_name in IMAGE_FILENAMES:
    count=count+1
    image=mp.Image.create_from_file(image_file_name)
    recognition_result=recognizer.recognize(image)                   #  Recognition_result=====>>>>    for every image
    
    images.append(image)
    print("HERE IS RESULT",count)
    print("\n")
    print(recognition_result)
    if (len(recognition_result.gestures) == 0):
        continue
    top_gesture=recognition_result.gestures[0][0]
    hand_landmarks=recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))
  
print("HERE IS IMAGES",images)
display_batch_of_images_with_gestures_and_hand_landmarks(images, results)



# visualization


import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.detections.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

with ObjectDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
  cap=cv2.VideoCapture(0)
  while True: 
     success, frame = cap.read()
     if not success: 
        break
     mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
     detector.detect_async(mp_image, 1000)
     image_copy = np.copy(image.numpy_view())
     annotated_image = visualize(image_copy, detection_result)
     rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
     cv2_imshow(rgb_annotated_image)
     
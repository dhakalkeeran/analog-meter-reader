# Analog Meter Reader
An application that reads an analog meter. This application uses an object detection model, [**YOLOv7**](https://github.com/WongKinYiu/yolov7.git), for localizing the analog meter and uses opencv for image processing algorithms to detect the needle in the meter. Finally, a postprocessing procedure is carried out to determine the reading on the analog meter.

![Analog meter needle detection image](/output_images/analog_meter_0.png "Analog meter needle detection")

## Inference
This inference code reuses some components from the official **YOLOv7** repository. 
The components help make the prediction using only the files in this repository and without any dependencies on the other files from the official repository.

Command to run the detection script:
```
python detection.py -i <image_folder>/<image_name> -v0 <value_at_0_degree> -v1 <value_at_180_degree> -f <boolean_value_to_write_result_in_file> 
```

For example:
```
python detection.py -i meter_images/analog_meter_0.png -v0 3.25 -v1 0.6 -f True 
```

Output:
```
reading: -0.04730729493180963, final_angle: 223.96804267461349
```

## Evaluation
Command to run the evaluation script:
```
python test_results.py 
```
For evaluation script, [label.txt](label.txt) has the meter readings at 0 and 180 degrees for the analog meters in the images.

The error report will be saved in [test_results.txt](test_results.txt).

## Object Detection Model
An object detection model is used to detect an analog meter in the image. The object detection is performed using the [**YOLOv7**](https://github.com/WongKinYiu/yolov7.git) model. The fine-tuning of the model was done as specified in the official repository.

To know more on how this model was trained and evaluated, please read the details in [report.pdf](report.pdf).

## Image Processing
After the analog meter is localized in the image using the **YOLOv7** model, image processing algorithms such as gaussian blur, canny edge detection, and hough transformation are performed in the cropped image to detect the pointer needle in the meter. 

Once the needle is detected, the slope of the needle is calculated. The slope along with the meter readings at 0 and 180 degrees are used to determine the final meter reading.

## Documentation
The whole project is documented in [analog_meter_reader.pdf](analog_meter_reader.pdf).
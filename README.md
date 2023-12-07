# Analog Meter Reader
An application that reads an analog meter. This application uses an object detection model, [**YOLOv7**](https://github.com/WongKinYiu/yolov7.git), for localizing the analog meter and uses opencv for image processing algorithms. Finally, a postprocessing procedure is carried out to determine the reading on the analog meter.

## Inference
This inference code reuses some components from the official **YOLOv7** repository. 
The components helps make the prediction using only the files in this repository and without any dependencies on the other files from the repository.

Command to run the program file:
```
python detection.py -i <image_folder>/<image_name> -v0 <value_at_0_degree> -v1 
<value_at_180_degree> -f <boolean_value_to_write_result_in_file> 
```

For example:
```
python detection.py -i meter_images/analog_meter_0.png -v0 3.25 -v1 0.6 -f True 
```

Command to run the evaluation script:
```
python test_results.py 
```

The error report will be saved in [**test_results.txt**](test_results.txt).

The test_results.txt file generated after running the program for a set of 30 images along with 
the label.txt file will be added along with this report. Additionally, sample inputs and output 
files will also be provided.

## Training
Training was performed using the official [**YOLOv7**](https://github.com/WongKinYiu/yolov7.git) repository. The procedure to follow for fine-tuning the model is clearly specified in the official repository.

To know more on how this model was trained and evaluated, please read the details on [report.pdf](report.pdf).

## Documentation
The whole project is documented at [analog_meter_reader.pdf](analog_meter_reader.pdf).
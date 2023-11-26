import os
import math

error = 0
n = 0

write_file = open("test_results.txt", "w")
write_file.write("name, true_reading, read_value, read_angle, deviation\n")

with open("label.txt", "r") as file:
    for line in file.readlines():
        values = [v.strip() for v in line.split(",")]
        name, val_180, val_0, reading = values
        
        val_180 = float(val_180)
        val_0 = float(val_0)
        reading = float(reading)
        print(name, val_180, val_0, reading)
        
        os.popen(f"python detection.py -i meter_images/{name}.png -v1 {val_180} -v0 {val_0} -f 1").read()
        with open("reading.txt", "r") as read_file:
            values = read_file.read().split(",")
            read_value, read_angle = float(values[0]), float(values[1])
            
        print(read_value, reading)
        
        write_file.write(f"{name}, {reading}, {read_value}, {read_angle}, {abs(reading-read_value)}\n")
        
        error += (read_value - reading)**2
        n += 1

rmse = math.sqrt(error/n)
write_file.write(f"\nRoot Mean Squared Error: {rmse}\n")
write_file.close()
print("RMSE:", rmse)
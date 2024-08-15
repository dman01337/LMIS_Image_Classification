import os
import sys
import numpy as np
import pandas as pd
import shutil
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Define the source directory and target size
source_dir = 'C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/Data/Raw/images_sorted_multiclass/Script_source'
dest_dir = 'C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/Data/Processed'


def get_truncated_path(path, max_length=100):
    if len(path) > max_length:
        path = '...' + path[-max_length:]
    return path

# Define a function to open a folder dialog
def open_folder_dialog(label, button_pressed):
    folder_selected = filedialog.askdirectory()
    if button_pressed == 'source':
        global source_dir
        source_dir = folder_selected
    elif button_pressed == 'dest':
        global dest_dir
        dest_dir = folder_selected
    # if folder_selected is too long then truncate the beginning and add '...'
    folder_selected = get_truncated_path(folder_selected)
    label.config(text=f"{folder_selected}")

# Define a function to print a progress bar for the file transfers
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '>' + '-' * (length - filled_length)
    print(f'\r{prefix} [{bar}] {percent}% {suffix}', end='\r')
    if iteration == total:
        bar = fill * filled_length
        print(f'\r{prefix} [{bar}] {percent}% {suffix}')

def classify_images(msg_labels):
    print('-'*50)
    msg_labels[0].config(text='Classifying Images (Please wait)...')
    msg_labels[0].update_idletasks()
    print('Classifying Images...')
    # Define the target size for the images
    target_size = (565, 614)  

    # Create an ImageDataGenerator for the source images
    datagen = ImageDataGenerator(rescale=1./65535)

    # Create a generator for the source images
    source_generator = datagen.flow_from_directory(
        source_dir,
        target_size=target_size,
        batch_size=1,
        shuffle=False,
        color_mode='grayscale',
        class_mode=None,
    )


    # Predict the classes of the images
    predictions = model.predict(source_generator)

    # Get the class indices
    class_indices = {0: 'CONTAMINATION', 1: 'DAMAGE', 2: 'PASS', 3: 'SPLIT'}
    class_counts = {class_name: 0 for class_name in class_indices.values()}

    # Convert predictions to class labels
    predicted_classes = [class_indices[np.argmax(pred)] for pred in predictions]

    # tally up the classes
    for filename, predicted_class in zip(source_generator.filenames, predicted_classes):
        class_counts[predicted_class] += 1

    # Print a message to indicate that the classification is complete
    print('Classification complete:')
    msg_labels[1].config(text='Classification complete:')
    msg_labels[1].update_idletasks()
    # print('-'*32)
    msg_labels[2].config(text='-'*37)
    msg_labels[2].update_idletasks()
    # print(f'| {"Class":^15} | {"Count":^10} |')
    msg_labels[3].config(text=f'| {"Class":^30} | {"Count":^15} |')
    msg_labels[3].update_idletasks()
    # print('-'*32)
    msg_labels[4].config(text='-'*37)
    msg_labels[4].update_idletasks()
    label_index = 5
    for class_name, count in class_counts.items():
        # print(f'| {class_name:^15} | {count:^10} |')
        if label_index == 5:
            spaces = 12
        elif label_index == 6:
            spaces = 29
        elif label_index == 7:
            spaces = 36
        elif label_index == 8:
            spaces = 36
        spaces = '.'*spaces
        msg_labels[label_index].config(text=f' {class_name}{spaces}{count}')
        msg_labels[label_index].update_idletasks()
        label_index += 1
    print('-'*32)
    print(f'| {"Total":^15} | {sum(class_counts.values()):^10} |')
    print('-'*32)

    # Create classification folders if they don't exist
    for class_name in class_indices.values():
        class_folder = os.path.join(dest_dir, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    # Check to see if the csv log file exists
    try:
        results_df = pd.read_csv(f'{dest_dir}/Classification_results.csv')   
    except:
        # Create a new dataframe
        results_df = pd.DataFrame(columns=['Timestamp', 'SN', 'Source_File', 'Dest_File', 'Predicted_Class'])

    print('Moving images to classification folders...')
    # Move images to their respective classification folders
    src_path = os.path.join(source_dir, 'Unclassified')
    num_files = len(source_generator.filenames)
    file_counter = 0
    for filename, predicted_class in zip(source_generator.filenames, predicted_classes):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, predicted_class, os.path.basename(filename))
        shutil.copy(src_path, dest_path)
        # Make the serial number by removing .tif
        serial_num = os.path.basename(filename)
        serial_num = serial_num.replace('.tif', '')
        # Add the results to the dataframe
        new_row = pd.DataFrame([{'Timestamp': pd.Timestamp.now(), 
                                'SN': serial_num, 
                                'Source_File': filename, 
                                'Dest_File': dest_path,
                                'Predicted_Class': predicted_class
        }])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        # Print the progress bar
        file_counter += 1
        print_progress_bar(file_counter, 
                        num_files, 
                        prefix=f'{file_counter}/{num_files}', 
                        suffix='Complete', 
                        length=30,
                        fill='=')

    # Save the results to a csv file
    results_df.to_csv(f'{dest_dir}/Classification_results.csv', index=False)

    print(f'The images have been moved to classification folders at:')
    print(f'{dest_dir}')
    print('Results have been saved to the log file at:')
    print(f'{dest_dir}/Classification_results.csv')

# Get the path to the executable's directory
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model
model_path = os.path.join(application_path, 'Model_Multi_19.h5')

# Load the model
model = load_model(model_path)

# Create the main window
root = tk.Tk()
root.title("LMIS Image Classification")
root.geometry("800x450")

# Create a button that opens the folder dialog for the source directory
button_x = 30
button_y = 50
source_button = tk.Button(root, text="Source Image Folder", 
                          command=lambda: open_folder_dialog(source_label, 'source'))
source_button.pack(pady=20)
source_button.place(x=button_x, y=button_y)
# Create a label to display messages for the source folder
source_label = tk.Label(root, text=get_truncated_path(source_dir))
source_label.pack(pady=20)
source_label.place(x=button_x+130, y=button_y)

# Create a button that opens the folder dialog for the classification directory
dest_button = tk.Button(root, text="Classification Folder", 
                        command=lambda: open_folder_dialog(dest_label, 'dest'))
dest_button.pack(pady=20)
dest_button.place(x=button_x, y=button_y+50)
# Create a label to display messages for the classification folder
dest_label = tk.Label(root, text=get_truncated_path(dest_dir))
dest_label.pack(pady=20)
dest_label.place(x=button_x+130, y=button_y+50)

# Create the message labels
msg_labels = []
for i in range(10):
    y_pos = 100 + i*16
    msg_labels.append(tk.Label(root, text=f''))
    msg_labels[i].pack(pady=20)
    msg_labels[i].place(x=button_x+130, 
                        y=button_y+y_pos)

# Create a button that opens the folder dialog for the classification directory
run_button = tk.Button(root, text="Classify Images", 
                        command=lambda: classify_images(msg_labels))
run_button.pack(pady=20)
run_button.place(x=button_x, y=button_y+100)

# Run the application
root.mainloop()




import os
import sys
import numpy as np
import pandas as pd
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


def get_truncated_path(path, max_length=110):
    ellipsis_index = -2
    while len(path) > max_length:
        ellipsis_index -= 1
        folders = path.split('/')
        folders[ellipsis_index] = '...'
        path = '/'.join(folders)
    return path

# Define a function to open a folder dialog
def open_folder_dialog(label, button_pressed):
    folder_selected = filedialog.askdirectory()
    if button_pressed == 'source':
        global source_dir
        source_dir = folder_selected
        source_file_path = os.path.join(text_file_path, 'source_dir.txt')
        with open(source_file_path, 'w') as f:
            f.write(source_dir)
        print(f"Source Directory Saved: {source_file_path}")
    elif button_pressed == 'dest':
        global dest_dir
        dest_dir = folder_selected
        dest_file_path = os.path.join(text_file_path, 'dest_dir.txt')
        with open(dest_file_path, 'w') as f:
            f.write(dest_dir)
        print(f"Destination Directory Saved: {dest_file_path}")
    # if folder_selected is too long then truncate the beginning and add '...'
    folder_selected = get_truncated_path(folder_selected)
    label.config(text=f"{folder_selected}")
    label.update_idletasks()
 



def move_files_to_subfolder(source_dir, subfolder_name):
    # Create the subfolder path
    subfolder_path = os.path.join(source_dir, subfolder_name)
    
    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    
    # Move all files from the source directory to the subfolder
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Move the file to the subfolder
        shutil.move(file_path, os.path.join(subfolder_path, filename))

def show_help():
    help_msg = '''
    1. Select the source images folder by clicking the "Source Images Folder" button.\n
    2. Select the destination folder for the classified images by clicking the "Classified Images Folder (Destination)" button.\n
    3. Click the "Classify Images" button to classify the images in the source folder and move them to the destination folder. WARNING: this will move the images from the source folder to the destination folder.\n
    4. The classified images can be reviewed in the destination folder within the subfolders for each class.
    '''
    messagebox.showinfo("Help", help_msg)

def classify_images(msg_labels, tree):
    # Clear the message labels
    for label in msg_labels:
        label.config(text=' '*200)
        label.update_idletasks()
    # Clear the Treeview widget
    for item in tree.get_children():
        tree.delete(item)
    tree.update_idletasks()

    # Define the target size for the images
    target_size = (330, 358)

    # check to see if the source directory exists
    if not os.path.exists(source_dir):
        msg_labels[0].config(text='Error: The source directory does not exist. Please select a valid destination directory and try again.')
        msg_labels[0].update_idletasks()
        return
    elif not os.path.exists(dest_dir):
        msg_labels[0].config(text='Error: The destination directory does not exist. Please select a valid destination directory and try again.')
        msg_labels[0].update_idletasks()
        return
    
    try:
        if len(os.listdir(source_dir)) != 0:
            move_files_to_subfolder(source_dir, 'temp')
    except:
        msg_labels[0].config(text='Error: Unable to classify images. Please check the source directory and try again.')
        msg_labels[0].update_idletasks()
        return
    
    # check to see if the destination directory exists


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

    # get the number of images found
    num_images = len(source_generator.filenames)
    if num_images == 0:
        msg_labels[0].config(text='Error: No images found in the source directory. Please check the source directory and try again.')
        msg_labels[0].update_idletasks()
        return
    else:
        msg_labels[0].config(text=f'Classifying {num_images} images, please wait...                                                 ')
        msg_labels[0].update_idletasks()

    # Predict the classes of the images
    try:
        predictions = model.predict(source_generator)
    except Exception as e:
        msg_labels[0].config(text='Error: Unable to classify images. Please check the source directory and try again.')
        msg_labels[0].update_idletasks()
        error_msg = str(e)
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + '... (truncated)'
        msg_labels[1].config(text=f'Error message: {error_msg}')
        msg_labels[1].update_idletasks()
        msg_labels[2].config(text='If the error persists, please contact Engineering.')
        msg_labels[2].update_idletasks()
        return

    # Get the class indices
    class_indices = {0: 'CONTAMINATION', 1: 'DAMAGE', 2: 'ETCH', 3: 'PASS', 4: 'SPLIT'}
    class_counts = {class_name: 0 for class_name in class_indices.values()}

    # Convert predictions to class labels
    predicted_classes = [class_indices[np.argmax(pred)] for pred in predictions]

    # tally up the classes
    for filename, predicted_class in zip(source_generator.filenames, predicted_classes):
        class_counts[predicted_class] += 1

    # Print a message to indicate that the classification is complete
    msg_labels[0].config(text='Classification complete:                         ')
    msg_labels[0].update_idletasks()

    # Update the Treeview widget with the classification results
    for class_name, count in class_counts.items():
        tree.insert("", tk.END, values=(class_name, count))
    tree.insert("", tk.END, values=('----------', '--------'))
    tree.insert("", tk.END, values=('Total Images', num_images))
    tree.update_idletasks()

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

    # Message to indicate that the images are being moved to the classification folders
    msg_labels[1].config(text=f'Moving {num_images} images to classification folders...')
    msg_labels[1].update_idletasks()
    # Move images to their respective classification folders
    src_path = os.path.join(source_dir, 'Unclassified')
    for filename, predicted_class in zip(source_generator.filenames, predicted_classes):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, predicted_class, os.path.basename(filename))
        try:
            shutil.move(src_path, dest_path)
        except:
            msg_labels[1].config(text='Error: Unable to move images to classification folders. Please check the source directory and try again.')
            msg_labels[1].update_idletasks()
            return
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

    # Save the results to a csv file
    results_df.to_csv(f'{dest_dir}/Classification_results.csv', index=False)

    # Message to indicate that the images have been moved to the classification folders
    msg_labels[1].config(text='The images have been moved to the classification folders in the destination directory listed above.')
    msg_labels[1].update_idletasks()
    msg_labels[2].config(text='Results have been saved to the csv file in the destination directory listed above.')
    msg_labels[2].update_idletasks()

    # delete the temp folder if it is empty
    if len(os.listdir(os.path.join(source_dir, 'temp'))) == 0:
        shutil.rmtree(os.path.join(source_dir, 'temp'))
    
# Get the path to the executable's directory
if getattr(sys, 'frozen', False):
    text_file_path = os.path.dirname(sys.executable)
else:
    text_file_path = os.path.dirname(os.path.abspath(__file__))

# Get the path to the executable's directory
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# load the previously used source and destination directories
try:
    with open(os.path.join(text_file_path, 'source_dir.txt'), 'r') as f:
        source_dir = f.read()
except:
    source_dir = '<-- Please select a source directory'
try:
    with open(os.path.join(text_file_path, 'dest_dir.txt'), 'r') as f:
        dest_dir = f.read()
except:
    dest_dir = '<-- Please select a destination directory'


# Define the source directory and target size
# source_dir = 'C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/Data/Raw/images_sorted_multiclass/Script_source'
# dest_dir = 'C:/Users/daled/Documents/Flatiron/Course_material/Phase_5/LMIS_Image_Classification/Data/Processed'

# Define the path to the model
model_path = os.path.join(application_path, 'Model_Multi_21.h5')

# Load the model
model = load_model(model_path)

# Create the main window
root = tk.Tk()
root.title("LMIS Image Classification")
root.geometry("800x450")
root.configure(bg="lightgray")

# Set the window icon using the .ico file
icon_path = os.path.join(application_path, 'eye.ico')
root.iconbitmap(icon_path)


# Create a button that opens the folder dialog for the source directory
button_x = 30
button_y = 30
source_button = tk.Button(root, text="Source Images Folder", 
                          command=lambda: open_folder_dialog(source_label, 'source'))
source_button.pack(pady=20)
source_button.place(x=button_x, y=button_y)
# Create a label to display messages for the source folder
source_label = tk.Label(root, text=get_truncated_path(source_dir), bg="lightgray")
source_label.pack(pady=20)
source_label.place(x=button_x+130, y=button_y)

# Create a button that opens the folder dialog for the classification directory
dest_button = tk.Button(root, text="Classified Images Folder (Destination)", wraplength=110,
                        command=lambda: open_folder_dialog(dest_label, 'dest'))
dest_button.pack(pady=20)
dest_button.place(x=button_x, y=button_y+40, height=45, width=125)
# Create a label to display messages for the classification folder
dest_label = tk.Label(root, text=get_truncated_path(dest_dir), bg="lightgray")
dest_label.pack(pady=20)
dest_label.place(x=button_x+130, y=button_y+50)

# Create the message labels
msg_labels = []
msg_labels.append(tk.Label(root, text='<-- If folders listed above are correct, click the button to classify images'))
msg_labels[0].pack(pady=20)
msg_labels[0].place(x=button_x+130, y=button_y+100)
msg_labels.append(tk.Label(root, text=''))
msg_labels[1].pack(pady=20)
msg_labels[1].place(x=button_x+130, y=button_y+330)
msg_labels.append(tk.Label(root, text=''))
msg_labels[2].pack(pady=20)
msg_labels[2].place(x=button_x+130, y=button_y+360)
for label in msg_labels:
    label.config(bg="lightgray")
    label.update_idletasks()

# Create a style for the Treeview
style = ttk.Style()
style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"), background="lightblue", foreground="black")

# Create a Treeview widget
tree = ttk.Treeview(root, 
                    columns=("Column1", "Column2"), 
                    show="headings")
tree.heading("Column1", text="Image Class")
tree.heading("Column2", text="Count")

tree.column("Column1", width=200, anchor="center")
tree.column("Column2", width=75, anchor="center")

# Pack the Treeview widget
tree.pack(pady=20)
tree.place(x=button_x+140, y=button_y+130, height=175)

# Create a button that opens the folder dialog for the classification directory
run_button = tk.Button(root, text="Classify Images", 
                        command=lambda: classify_images(msg_labels, tree))
run_button.pack(pady=20)
run_button.place(x=button_x, y=button_y+100, width=125)

# Create a menu bar for the help menu
menu_bar = tk.Menu(root)
# Create a Help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="Help", command=show_help)
# Add the Help menu to the menu bar
menu_bar.add_cascade(label="Help", menu=help_menu)
# Configure the window to use the menu bar
root.config(menu=menu_bar)

# Run the application
root.mainloop()




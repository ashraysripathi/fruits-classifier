
DATASET_PATH=""
dl=input("Have you downloaded the necessary dataset?(Y/N)")
if dl=='N' or dl=='n':
    import extract
    extract
    DATASET_PATH=extract.folder_selected
else:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    from tkinter import *
    print("Select the folder where dataset is located")
    root = Tk()
    root.withdraw()
    DATASET_PATH = filedialog.askdirectory()
    TRAIN_DIR=DATASET_PATH+"\Training"

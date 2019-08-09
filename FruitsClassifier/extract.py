import kaggle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import *
import time

print("Authenticating......")
kaggle.api.authenticate()
print("Authenticating Complete")
print("Choose folder to download and extract dataset to...")
time.sleep(1)   #so people can read
root = Tk()
root.withdraw()
folder_selected= filedialog.askdirectory()
folder_selected=folder_selected.replace('\\','\\\\')
print(folder_selected)
print("Downloading please wait............")
try:
    kaggle.api.dataset_download_files('moltean/fruits', path=folder_selected, unzip=True)
    print("Extraction Complete")
except:
    print("No Internet Connection")

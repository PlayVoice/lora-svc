import os
import glob

# List of file extensions to delete
extensions = [".wav", ".npy", ".csv", ".mp4", ".mp3"]

# Loop through each extension
for ext in extensions:
  # Find all files with that extension in the current directory
  files = glob.glob("*" + ext)
  # Loop through each file and delete it
  for file in files:
    os.remove(file)
    print("Deleted", file)

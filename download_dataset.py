import kagglehub
import os

# The dataset will be downloaded to a cache folder by default.
# We can find its location and then work with the files from there.
path = kagglehub.dataset_download("isuruai/traffic-light-signals-at-various-intersections")

print(f"Dataset downloaded to: {path}")

# Let's list the files to confirm a successful download
print("\nFiles in the dataset directory:")
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 
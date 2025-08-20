import kagglehub

# Download latest version
path = kagglehub.dataset_download("avc0706/luna16")

print("Path to dataset files:", path)

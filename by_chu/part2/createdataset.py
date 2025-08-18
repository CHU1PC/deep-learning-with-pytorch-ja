import kagglehub

# Download latest version
path = kagglehub.dataset_download("eliasmarcon/luna-16")

print("Path to dataset files:", path)

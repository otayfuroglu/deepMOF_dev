config = {"batch_size": 1, "cutoff_radius": 6.0,
          "n_radial": 20, "n_angular":5, "mode": "Behler"}

print()
print("=" * 30)
print("Model configuration parameters:")
for key, value in config.items():
    print("%s -> %s" %(key, value))
print("=" * 30)
print()

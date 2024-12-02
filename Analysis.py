import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file = "P01_Control_NV_Side.csv"  # Replace with your CSV file name
df = pd.read_csv(csv_file)

# Filter the frames between 2360 and 2560
start_frame = 2360
end_frame = 2560
filtered_df = df.loc[start_frame:end_frame, ["left_wrist_y"]]

# Plot the column 'left_wrist_y'
plt.figure(figsize=(10, 5))
plt.plot(filtered_df.index, 1-filtered_df["left_wrist_y"], label="Left Wrist Y", marker="o")

# Customize the plot
plt.title("Left Wrist Y Position (Frames 2360 to 2560)")
plt.xlabel("Frame Number")
plt.ylabel("Left Wrist Y")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

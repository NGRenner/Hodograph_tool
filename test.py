import imageio.v2 as imageio

# === USER SETTINGS ===
input_gif = "combined_plot_rap_20240526_1200_to_20240527_0000.gif"  # your existing GIF
output_gif = "combined_plot_slower.gif"  # output file name
new_duration = 1.5  # seconds per frame (adjust this)

# === LOAD AND REWRITE ===
gif = imageio.mimread(input_gif)  # Read all frames
imageio.mimsave(output_gif, gif, duration=new_duration, loop=0)  # Save with new speed
print(f"✅ Saved slower GIF to: {output_gif}")
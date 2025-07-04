import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import pandas as pd
from pathlib import Path
import random
import math

# Configuration section
config = {
    "csv_file": Path(' '),  # CSV file path
    "image_column": "img_path",  # Image path column name
    "output_csv": Path(' '),  # Output CSV file path
    "rectangle_size": 224,  # Box size in pixels (square side length)
    "title_format": 'Pixel Coordinates: ({x}, {y})'  # Title format
}


def load_image(image_path):

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return mpimg.imread(str(image_path))


def get_mouse_coordinates(image_path, rectangle_size, title_format):


    img = load_image(image_path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Move the mouse over the image")

    img_height, img_width = img.shape[:2]

    rect = Rectangle((0, 0), rectangle_size, rectangle_size,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    def update_rectangle(x, y):
        """Update rectangle position to center it at (x, y)."""
        half_size = rectangle_size / 2
        lower_left_x = max(x - half_size, 0)
        lower_left_y = max(y - half_size, 0)
        lower_left_x = min(lower_left_x, img_width - rectangle_size)
        lower_left_y = min(lower_left_y, img_height - rectangle_size)
        rect.set_xy((lower_left_x, lower_left_y))
        fig.canvas.draw_idle()

    coordinates = []

    def on_mouse_click(event):
        if event.inaxes and event.button == 1:  # Left mouse click
            x, y = int(event.xdata), int(event.ydata)
            coordinates.extend([x, y])
            print(f'Clicked at Pixel Coordinates: ({x}, {y})')
            plt.close(fig)

    fig.canvas.mpl_connect('motion_notify_event', lambda e: update_rectangle(int(e.xdata), int(e.ydata)) if e.inaxes else None)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    plt.show()

    return coordinates


def confirm_selection(avg_x, avg_y):

    while True:
        print(f"\nAverage Coordinates: ({avg_x}, {avg_y}). Choose an option:")
        print("1: Use the calculated average coordinates")
        print("2: Enter nozzle_tip_x and nozzle_tip_y manually")
        print("3: Re-select three points")
        print("4: Exit the program")

        user_input = input("Enter your choice (1/2/3/4): ").strip()

        if user_input == "1":
            return avg_x, avg_y
        elif user_input == "2":
            try:
                new_x = int(input("Enter nozzle_tip_x: ").strip())
                new_y = int(input("Enter nozzle_tip_y: ").strip())
                print(f"Manually entered coordinates: ({new_x}, {new_y})")
                return new_x, new_y
            except ValueError:
                print("Invalid input. Please enter integers for coordinates.")
        elif user_input == "3":
            print("Re-selecting three points...")
            return None, None
        elif user_input == "4":
            print("Exiting the program. Goodbye!")
            exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def main(config):
    csv_file = config.get("csv_file")
    df = pd.read_csv(csv_file)


    base_dir = csv_file.parents[1]

    image_column = config.get("image_column")
    rectangle_size = config.get("rectangle_size", 224)
    title_format = config.get("title_format", 'Pixel Coordinates: ({x}, {y})')

    while True:

        selected_images = df.sample(n=3, random_state=None)

        coordinates = []

        for _, row in selected_images.iterrows():

            full_image_path = base_dir / Path(row[image_column])

            print(f"Processing image: {full_image_path}")
            try:
                coordinates.append(get_mouse_coordinates(full_image_path, rectangle_size, title_format))
            except FileNotFoundError as e:
                print(e)
                return

        avg_x = int(sum(coord[0] for coord in coordinates) / len(coordinates))
        avg_y = int(sum(coord[1] for coord in coordinates) / len(coordinates))
        print(f"Computed Average Coordinates: ({avg_x}, {avg_y})")

        result = confirm_selection(avg_x, avg_y)
        if result == (None, None):
            continue
        else:
            final_x, final_y = result
            print(f"Final Coordinates: ({final_x}, {final_y})")
            break

    # Update CSV file
    df['nozzle_tip_x'] = final_x
    df['nozzle_tip_y'] = final_y
    output_csv = config.get("output_csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

    print(f"Final Coordinates (before random perturbation): ({final_x}, {final_y})")

    angle = random.uniform(0, 2 * math.pi)
    radius = 5 * math.sqrt(random.uniform(0, 1))
    perturb_x = int(radius * math.cos(angle))
    perturb_y = int(radius * math.sin(angle))
    random_center_x = final_x + perturb_x
    random_center_y = final_y + perturb_y

    print(f"Randomized Center within radius 5: ({random_center_x}, {random_center_y})")

    df['nozzle_tip_x'] = random_center_x
    df['nozzle_tip_y'] = random_center_y
    output_csv = config.get("output_csv")
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")


if __name__ == "__main__":
    main(config)
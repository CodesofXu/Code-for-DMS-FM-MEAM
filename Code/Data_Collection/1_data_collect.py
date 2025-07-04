import os
import time
import random
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
import csv
import queue
from datetime import datetime
from PIL import Image, ImageTk
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.keys import Keys

config = {
    "driver_path": 'D:/WebDriver/msedgedriver.exe',  # Replace with your msedgedriver path
    "url": 'https://www.baidu.com',  # Target webpage URL
    "input_box_id": 'kw',  # Input box element ID
    "change_interval": 300,  # Interval between parameter changes (seconds)
    "no_save_duration": 60,  # Duration to pause saving after parameter change (seconds)
    "camera_index": 0,       # Webcam device index
    "save_frequency_hz": 5,  # Image capture frequency (Hz)
    "random_mode": "pure"    # Random number generation method: "pure" or "pseudo"
}


driver = None
collecting_images = False
changing_extrusion = False
stop_threads = False
random_number = None
last_random_change = time.time()
last_change_time = 0
image_queue = queue.Queue()



def generate_random_number():
    global random_number
    if config["random_mode"] == "pure":
        random_number = random.choice(range(50, 151, 10))  # Pure random selection
    elif config["random_mode"] == "pseudo":
        # Pseudo-random generation logic
        if random_number is None:
            random_number = 50  # Initial value
        random_number = (random_number * 1103515245 + 12345) % 101  # Pseudo-random algorithm
        random_number = 50 + (random_number % 11) * 10  # Ensure value between 50-150



def open_browser():
    global driver
    service = Service(executable_path=config["driver_path"])
    options = webdriver.EdgeOptions()
    driver = webdriver.Edge(service=service, options=options)
    driver.get(config["url"])
    print("Browser launched")



def close_browser():
    global driver
    if driver:
        driver.quit()
        driver = None
    print("Browser closed")



def collect_images(cap):
    global collecting_images, stop_threads
    while not stop_threads:
        if collecting_images:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1080, 720))
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                image_queue.put((frame, random_number, timestamp))
            time.sleep(1 / config["save_frequency_hz"])
        else:
            time.sleep(0.5)  # Sleep to reduce CPU usage



def save_images():
    global stop_threads, last_change_time
    folder_index = 1
    while os.path.exists(f'imagedata{folder_index}'):
        folder_index += 1
    os.makedirs(f'imagedata{folder_index}')
    with open(f'imagedata{folder_index}/data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['index', 'img_path', 'flow_rate', 'timestamp'])
        img_index = 1
        while not stop_threads:
            if collecting_images and not image_queue.empty() and (
                    time.time() - last_change_time >= config["no_save_duration"]):
                frame, flow_rate, timestamp = image_queue.get()
                img_path = f'imagedata{folder_index}/image_{img_index}.jpg'
                cv2.imwrite(img_path, frame)
                csv_writer.writerow([img_index, img_path, flow_rate, timestamp])
                csv_file.flush()
                print(f"Saved image: {img_path}, Flow rate: {flow_rate}, Timestamp: {timestamp}")
                img_index += 1
            else:
                time.sleep(0.5)  # Sleep to reduce CPU usage



def change_extrusion():
    global random_number, last_random_change, stop_threads, changing_extrusion, last_change_time
    while not stop_threads:
        if changing_extrusion and time.time() - last_random_change >= config["change_interval"]:
            generate_random_number()  # Generate new random value
            last_random_change = time.time()
            last_change_time = time.time()
            if driver:
                try:
                    input_box = driver.find_element(By.ID, config["input_box_id"])
                    input_box.clear()
                    input_box.send_keys(f'M221 S{random_number}')
                    input_box.send_keys(Keys.RETURN)
                except Exception as e:
                    print(f"Error occurred: {e}")
            time.sleep(config["no_save_duration"])  # Pause saving after change
        time.sleep(1)



def start_collecting():
    global collecting_images
    collecting_images = True
    print("Image collection started")



def pause_collecting():
    global collecting_images
    collecting_images = False
    print("Image collection paused")



def start_changing_extrusion():
    global changing_extrusion, last_random_change, random_number
    if random_number is None:
        generate_random_number()
    changing_extrusion = True
    last_random_change = time.time()
    last_change_time = time.time()
    print("Extrusion parameter modification started")
    if driver:
        try:
            input_box = driver.find_element(By.ID, config["input_box_id"])
            input_box.clear()
            input_box.send_keys(f'M221 S{random_number}')
            input_box.send_keys(Keys.RETURN)
        except Exception as e:
            print(f"Error occurred: {e}")



def stop_changing_extrusion():
    global changing_extrusion
    changing_extrusion = False
    print("Extrusion parameter modification stopped")



def update_input_box_id():
    new_id = input_box_id_entry.get()
    if new_id:
        config["input_box_id"] = new_id
        print(f"Input box ID updated to: {new_id}")
        try:
            if driver:
                input_box = driver.find_element(By.ID, config["input_box_id"])
                input_box.clear()
                input_box.send_keys("")
                print("Input box ID validated successfully.")
            else:
                print("Browser not open, validation skipped.")
        except Exception as e:
            print(f"Invalid input box ID: {e}")
            pause_collecting()
            stop_changing_extrusion()



def update_status_label():
    next_change_time = max(0, config["change_interval"] - (time.time() - last_random_change))
    status_message = (
        f"Current extrusion rate: {random_number}\n"
        f"Next change in: {next_change_time:.1f} seconds\n"
        f"Image saving: {'Active' if collecting_images else 'Paused'}"
    )
    status_label.config(text=status_message)



def toggle_button_state(button, is_active):
    if is_active:
        button.config(state='normal')
    else:
        button.config(state='disabled')



def update_button_state():
    toggle_button_state(collecting_button, not collecting_images)
    toggle_button_state(change_button, not changing_extrusion)



def close_program(root, cap):
    global stop_threads, driver
    print("Terminating program")
    stop_threads = True
    cap.release()
    cv2.destroyAllWindows()
    if driver:
        driver.quit()
    root.quit()



def update_frame(cap, lbl):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (1080, 720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
    if not stop_threads:
        lbl.after(10, lambda: update_frame(cap, lbl))



def create_gui():
    global root, input_box_id_entry, status_label, collecting_button, change_button
    root = tk.Tk()
    root.title("Image Acquisition Control")

    window_width = 1000
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    root.config(bg="#f0f0f0")

    frame = ttk.Frame(root, padding="20")
    frame.grid(row=0, column=0, sticky='nsew')

    buttons = [
        ("Start Collection", start_collecting, "green"),
        ("Pause Collection", pause_collecting, "red"),
        ("Start Parameter Change", start_changing_extrusion, "blue"),
        ("Stop Parameter Change", stop_changing_extrusion, "orange"),
        ("Open Browser", open_browser, "purple"),
        ("Close Browser", close_browser, "brown"),
    ]
    for text, command, color in buttons:
        btn = ttk.Button(frame, text=text, command=command, width=12)
        btn.grid(sticky='ew', pady=5)
        if text == "Start Collection":
            collecting_button = btn
        if text == "Start Parameter Change":
            change_button = btn

    ttk.Label(frame, text="Input Box ID:").grid(sticky="w", pady=5)
    input_box_id_entry = ttk.Entry(frame)
    input_box_id_entry.grid(sticky='ew', pady=5)
    ttk.Button(frame, text="Update ID", command=update_input_box_id).grid(sticky='ew', pady=5)


    status_label = ttk.Label(frame, text="Status: Ready", anchor="w", width=50)
    status_label.grid(sticky='w', pady=5, row=7, column=0, rowspan=3)


    lbl = ttk.Label(root)
    lbl.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


    global cap
    cap = cv2.VideoCapture(config["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    threading.Thread(target=collect_images, args=(cap,), daemon=True).start()
    threading.Thread(target=save_images, daemon=True).start()
    threading.Thread(target=change_extrusion, daemon=True).start()


    update_frame(cap, lbl)


    def status_updater():
        update_status_label()
        update_button_state()
        root.after(1000, status_updater)

    status_updater()

    root.mainloop()



if __name__ == "__main__":
    create_gui()
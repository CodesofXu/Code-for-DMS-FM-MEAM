

import cv2
from tkinter import Tk, Canvas, Button, Label, Frame, messagebox, Entry, StringVar
from PIL import Image, ImageTk
import torch  # Import PyTorch
from torchvision import transforms
import threading
import time
import numpy as np
from datetime import datetime
from queue import Queue
from collections import defaultdict, deque
from threading import Lock, Event
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import os
import yaml  # Import PyYAML module
import logging
from control_algorithm_v4 import ControlAlgorithmFactory, ControlAlgorithmBase  # Import control algorithm factory function and base class

def load_config(config_path=' '):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise ValueError(f"Configuration file parsing error: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    defaults = {
        "control_algorithm": "Fuzzy",
        "set_point": 100.0,
        "Kp": 1.0,
        "Ki": 0,
        "Kd": 0,
        "input_interval_seconds": 10,
        "Custom1_param": 1.0,
        "output_p_field_id": "output_p_input"

    }

    required_keys = [
        "center_x", "center_y", "width", "height",
        "inference_frequency", "display_frequency",
        "model_path", "browser_url", "driver_path",
        "input_field_id",
        "category_mapping",
        "output_p_field_id"
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required keys in configuration file: {', '.join(missing_keys)}")

    for key, value in defaults.items():
        config.setdefault(key, value)

    config["inference_deque_maxlen"] = int(config["inference_frequency"] * config["input_interval_seconds"])

    if config["control_algorithm"] not in ["PID", "Fuzzy", "Bang_Bang"]:
        logging.warning("Control algorithm not in predefined list. Using Fuzzy control algorithm as default.")
        config["control_algorithm"] = "Fuzzy"

    int_keys = ["center_x", "center_y", "width", "height", "inference_frequency", "display_frequency",
                "inference_deque_maxlen"]
    float_keys = ["Kp", "Ki", "Kd", "set_point", "input_interval_seconds", "Custom1_param"]
    for key in int_keys:
        if not isinstance(config[key], int):
            raise TypeError(f"Key '{key}' in configuration file should be integer type.")
    for key in float_keys:
        if not isinstance(config[key], (int, float)):
            raise TypeError(f"Key '{key}' in configuration file should be float type.")

    if not isinstance(config["category_mapping"], dict):
        raise TypeError("'category_mapping' in configuration file should be dictionary type.")

    return config


config = load_config()

category_mapping = config["category_mapping"]

model = torch.jit.load(config["model_path"])
model.eval()
logging.info("TorchScript regression model loaded.")

control_algorithm_params = {}
if config["control_algorithm"] == "PID":
    control_algorithm_params = {
        "Kp": config.get("Kp", 1.0),
        "Ki": config.get("Ki", 0 ),
        "Kd": config.get("Kd", 0 ),
        "set_point": config.get("set_point", 100)
    }
elif config["control_algorithm"] == "Fuzzy":
    control_algorithm_params = {
        "set_point": config.get("set_point", 100)
    }
elif config["control_algorithm"] == "Bang_Bang":
    control_algorithm_params = {
        "set_point": config.get("set_point", 100)
    }
control_algorithm = ControlAlgorithmFactory(
    algorithm_name=config["control_algorithm"],
    **control_algorithm_params
)
logging.info(f"Control algorithm '{config['control_algorithm']}' initialized.")

inference_deque_maxlen = int(config["inference_frequency"] * config["input_interval_seconds"])
inference_results_deque = deque(maxlen=inference_deque_maxlen)

frame_queue = Queue(maxsize=5)
current_inference_result = None
result_lock = Lock()
running = True
inference_running = False
inference_event = Event()
result_accumulator = defaultdict(float)
total_inference_time = 0.0

browser_driver = None
browser_running = False
browser_lock = Lock()

input_thread = None
input_thread_running = False
input_lock = Lock()

root = Tk()
root.title("Real-time Video - With Bounding Box and Inference")

input_field_id_var = StringVar(value=config.get("input_field_id", "input_field_id"))
output_p_field_id_var = StringVar(value=config.get("output_p_field_id", "output_p_field_id"))

def update_input_field_id(*args):
    new_id = input_field_id_var.get().strip()
    if new_id:
        config["input_field_id"] = new_id
        logging.info(f"input_field_id updated to: {new_id}")


def update_output_p_field_id(*args):
    new_id = output_p_field_id_var.get().strip()
    if new_id:
        config["output_p_field_id"] = new_id
        logging.info(f"output_p_field_id updated to: {new_id}")


input_field_id_var.trace_add("write", update_input_field_id)
output_p_field_id_var.trace_add("write", update_output_p_field_id)

def create_control_inputs(control_frame, input_field_id_var, output_p_field_id_var):

    input_id_label = Label(control_frame, text="Input Field ID:", font=("Arial", 12), bg="lightgray")
    input_id_label.pack(pady=(10, 0))

    input_field_id_entry = Entry(control_frame, textvariable=input_field_id_var, width=20, font=("Arial", 12))
    input_field_id_entry.pack(pady=5)

    # Output P Field ID
    output_p_id_label = Label(control_frame, text="Output P Field ID:", font=("Arial", 12), bg="lightgray")
    output_p_id_label.pack(pady=(10, 0))

    output_p_field_id_entry = Entry(control_frame, textvariable=output_p_field_id_var, width=20, font=("Arial", 12))
    output_p_field_id_entry.pack(pady=5)


def toggle_inference():

    global inference_running, input_thread_running
    with result_lock:
        inference_running = not inference_running
    if inference_running:
        inference_event.set()
        toggle_button.config(text="Stop Inference", bg="#e74c3c", fg="white")
        status_label.config(text="Status: Running", fg="green")
        # Enable input button
        input_button.config(state="normal")
        logging.info("Inference started.")
    else:
        inference_event.clear()
        toggle_button.config(text="Start Inference", bg="#27ae60", fg="white")
        status_label.config(text="Status: Stopped", fg="red")
        # Disable input button and stop input thread
        input_button.config(state="disabled")
        if input_thread_running:
            toggle_input()  # This will stop the input thread
        logging.info("Inference stopped.")


def reset_accumulator():

    global current_inference_result, total_inference_time

    with result_lock:
        if inference_results_deque:
            average = sum(list(inference_results_deque)[-5:]) / 5
            current_inference_result = average
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"[{timestamp}] Current second average inference result: {average:.2f}, Total inference time: {total_inference_time:.1f}ms")
            # Reset inference time
            total_inference_time = 0.0
        else:
            current_inference_result = None
            logging.info("No inference results available for averaging.")


def inference_thread_func():

    global running, total_inference_time, current_inference_result

    last_inference_time = time.perf_counter()
    last_reset_time = time.perf_counter()

    while running:
        current_time = time.perf_counter()

        if current_time - last_reset_time >= 1.0:
            reset_accumulator()
            last_reset_time = current_time

        if inference_running and (current_time - last_inference_time >= 1 / config["inference_frequency"]):
            if not frame_queue.empty():

                while not frame_queue.empty():
                    frame = frame_queue.get()

                center_x, center_y, width, height = (
                    config["center_x"],
                    config["center_y"],
                    config["width"],
                    config["height"],
                )

                top_left = (center_x - width // 2, center_y - height // 2)
                bottom_right = (center_x + width // 2, center_y + height // 2)
                cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                try:

                    img_mean = config["imagenet_mean"]
                    img_std = config["imagenet_std"]

                    transform_sequence = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=img_mean, std=img_std)
                    ])

                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                    input_tensor = transform_sequence(cropped_pil)
                    input_tensor = input_tensor.unsqueeze(0)

                    start_time = time.perf_counter()
                    with torch.no_grad():
                        output = model(input_tensor)
                    inference_time = (time.perf_counter() - start_time) * 1000

                    extrusion_value = output.item()

                    with result_lock:
                        inference_results_deque.append(extrusion_value)

                        total_inference_time += inference_time

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        logging.info(f"[{timestamp}] Inference result: {extrusion_value:.2f}, Inference time: {inference_time:.1f}ms")
                except Exception as e:
                    logging.error(f"Inference thread error: {e}")

            last_inference_time = current_time
        else:
            inference_event.wait()


def update_frame():

    global running

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (1280, 720))

    if not ret:
        logging.error("Unable to read from camera!")
        running = False
        return

    if not frame_queue.full():
        frame_queue.put(frame)

    center_x, center_y, width, height = (
        config["center_x"],
        config["center_y"],
        config["width"],
        config["height"],
    )
    top_left = (center_x - width // 2, center_y - height // 2)
    bottom_right = (center_x + width // 2, center_y + height // 2)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    with result_lock:
        inference_display = current_inference_result

    if inference_display is not None:
        cv2.putText(
            frame,
            f"Inference Avg: {inference_display:.2f}",
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    if running:
        root.after(33, update_frame)


def toggle_browser():

    global browser_driver, browser_running, input_thread_running

    with browser_lock:
        if not browser_running:
            try:

                edge_options = EdgeOptions()
                edge_options.add_argument("--start-maximized")  # Maximize window on start

                edgedriver_path = config["driver_path"]
                if not os.path.exists(edgedriver_path):
                    raise FileNotFoundError(f"msedgedriver.exe not found: {edgedriver_path}")

                service = EdgeService(executable_path=edgedriver_path)
                browser_driver = webdriver.Edge(service=service, options=edge_options)

                browser_driver.get(config["browser_url"])

                browser_running = True
                browser_button.config(text="Close Browser", bg="#e74c3c", fg="white")
                browser_status_label.config(text="Browser Status: Open", fg="green")
                logging.info("Browser opened.")
            except Exception as e:
                logging.error(f"Unable to open browser: {e}")
        else:
            try:
                if browser_driver:
                    browser_driver.quit()
                    browser_driver = None

                browser_running = False
                browser_button.config(text="Open Browser", bg="#27ae60", fg="white")
                browser_status_label.config(text="Browser Status: Closed", fg="red")
                logging.info("Browser closed.")

                if input_thread_running:
                    toggle_input()
            except Exception as e:
                logging.error(f"Unable to close browser: {e}")


def toggle_input():

    global input_thread_running, input_thread

    with input_lock:
        if not input_thread_running:
            if not browser_running:
                messagebox.showwarning("Warning", "Please open browser before starting input operation.")
                return
            try:
                input_thread_running = True
                input_thread = threading.Thread(target=input_loop, daemon=True)
                input_thread.start()
                input_button.config(text="Stop Input", bg="#e74c3c", fg="white")
                input_status_label.config(text="Input Status: Running", fg="green")
                logging.info("Input operation started.")
            except Exception as e:
                logging.error(f"Unable to start input thread: {e}")
        else:
            try:
                input_thread_running = False
                input_button.config(text="Start Input", bg="#27ae60", fg="white")
                input_status_label.config(text="Input Status: Stopped", fg="red")
                logging.info("Input operation stopped.")
            except Exception as e:
                logging.error(f"Unable to stop input thread: {e}")


def input_loop():
    """Input thread: periodically input string to specified field"""
    while input_thread_running and running and browser_running:
        try:
            with result_lock:

                current_inference_results = list(inference_results_deque)


            extrusion_output = 0


            if config["control_algorithm"] == "Fuzzy":

                try:
                    target_label = "Extrusion Flow"
                    slider_xpath = f'''
                        //div[contains(@class, "text-body-1") and contains(text(), "{target_label}")]
                        /ancestor::form//div[@role="slider" and @aria-valuenow]
                    '''

                    wait = WebDriverWait(browser_driver, 10)
                    slider_element = wait.until(EC.visibility_of_element_located((By.XPATH, slider_xpath)))

                    output_p_value_str = slider_element.get_attribute("aria-valuenow")
                    if ControlAlgorithmBase.is_int(output_p_value_str):
                        output_p = int(output_p_value_str)
                        logging.info(f"Current extrusion flow: '{output_p}'")
                    else:
                        logging.warning(f"Read output_p value '{output_p_value_str}' is not a valid integer. Using default 0.")
                        output_p = 100
                except Exception as e:
                    logging.error(f"Unable to read output_p: {e}")
                    output_p = 0

                extrusion_output = control_algorithm.calculate_extrusion(current_inference_results, output_p=output_p, Boxplot=config["Boxplot"])
            elif config["control_algorithm"] == "PID":
                try:
                    target_label = "Extrusion Flow"
                    slider_xpath = f'''
                        //div[contains(@class, "text-body-1") and contains(text(), "{target_label}")]
                        /ancestor::form//div[@role="slider" and @aria-valuenow]
                    '''

                    wait = WebDriverWait(browser_driver, 10)
                    slider_element = wait.until(EC.visibility_of_element_located((By.XPATH, slider_xpath)))

                    output_p_value_str = slider_element.get_attribute("aria-valuenow")
                    if ControlAlgorithmBase.is_int(output_p_value_str):
                        output_p = int(output_p_value_str)
                        logging.info(f"Current extrusion flow: '{output_p}'")
                    else:
                        logging.warning(f"Read output_p value '{output_p_value_str}' is not a valid integer. Using default 0.")
                        output_p = 100
                except Exception as e:
                    logging.error(f"Unable to read output_p: {e}")
                    output_p = 0
                extrusion_output = control_algorithm.calculate_extrusion(current_inference_results, output_p=output_p,
                                                                         Boxplot=config["Boxplot"])
            else:

                extrusion_output = control_algorithm.calculate_extrusion(current_inference_results)

            dynamic_input_string = f"M221 S{extrusion_output:.0f}"

            input_xpath = '''
            //div[contains(@class, "col") and @data-v-b9bef6e6]
            //div[contains(@class, "v-text-field__slot")]/textarea
            '''

            wait = WebDriverWait(browser_driver, 10)
            input_element = wait.until(
                EC.presence_of_element_located((By.XPATH, input_xpath))
            )

            input_element.clear()

            input_element.send_keys(dynamic_input_string)
            input_element.send_keys(Keys.RETURN)
            logging.info(f"Input to field: {dynamic_input_string}")
        except Exception as e:
            logging.error(f"Input thread error: {e}")

        time.sleep(config["input_interval_seconds"])


def on_closing():

    global running, inference_running, browser_running, input_thread_running

    try:
        running = False
        inference_event.set()
        if inference_thread.is_alive():
            inference_thread.join()

        input_thread_running = False
        if input_thread and input_thread.is_alive():
            input_thread.join()

        if browser_running:
            toggle_browser()

        cap.release()
        logging.info("All resources released.")
    except Exception as e:
        logging.error(f"Error closing program: {e}")
    finally:
        root.destroy()


main_frame = Frame(root, bg="lightgray")
main_frame.pack(fill="both", expand=True)

video_frame = Frame(main_frame, bg="black")
video_frame.pack(side="left", fill="both", expand=True)

canvas = Canvas(video_frame, width=1280, height=720, bg="black")
canvas.pack()

control_frame = Frame(main_frame, bg="lightgray", width=200)
control_frame.pack(side="right", fill="y")

status_label = Label(control_frame, text="Status: Stopped", font=("Arial", 14), fg="red", bg="lightgray")
status_label.pack(pady=20)

toggle_button = Button(control_frame, text="Start Inference", command=toggle_inference, font=("Helvetica Neue", 14),
                       bg="#27ae60", fg="white", relief="flat", width=15, height=2)
toggle_button.pack(pady=10)

browser_status_label = Label(control_frame, text="Browser Status: Closed", font=("Arial", 14), fg="red", bg="lightgray")
browser_status_label.pack(pady=20)

browser_button = Button(control_frame, text="Open Browser", command=toggle_browser, font=("Helvetica Neue", 14),
                        bg="#27ae60", fg="white", relief="flat", width=15, height=2)
browser_button.pack(pady=10)

input_status_label = Label(control_frame, text="Input Status: Stopped", font=("Arial", 14), fg="red", bg="lightgray")
input_status_label.pack(pady=20)

input_button = Button(control_frame, text="Start Input", command=toggle_input, font=("Helvetica Neue", 14), bg="#27ae60",
                      fg="white", relief="flat", width=15, height=2, state="disabled")
input_button.pack(pady=10)

create_control_inputs(control_frame, input_field_id_var, output_p_field_id_var)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.critical("Unable to open camera!")
    raise RuntimeError("Unable to open camera!")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["center_x"] * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["center_y"] * 2)
cap.set(cv2.CAP_PROP_FPS, 30)
logging.info("Camera initialized.")

inference_thread = threading.Thread(target=inference_thread_func, daemon=True)
inference_thread.start()
logging.info("Inference thread started.")

update_frame()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()

running = False
inference_event.set()
input_thread_running = False
if browser_running:
    toggle_browser()
cap.release()
logging.info("Program exited.")
# control_algorithm_v6.py

import logging
from abc import ABC, abstractmethod
from typing import List
from logging.handlers import RotatingFileHandler
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from collections import Counter
import os
import glob

import time


# ========== Logging Configuration Start ==========

def setup_logging():

    # Define log directory
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    existing_logs = glob.glob(os.path.join(log_dir, "control_algorithm_v*.log"))
    existing_versions = [
        int(os.path.splitext(os.path.basename(log))[0].split('_v')[-1])
        for log in existing_logs
        if os.path.splitext(os.path.basename(log))[0].split('_v')[-1].isdigit()
    ]

    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 1

    log_filename = os.path.join(log_dir, f"control_algorithm_v{next_version}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=0
    )
    file_handler.setLevel(logging.DEBUG)  # Record DEBUG level and above in file
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Display INFO level and above in console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info(f"Log file created: {log_filename}")

setup_logging()





class ControlAlgorithmBase(ABC):


    @abstractmethod
    def calculate_extrusion(self, inference_results: List[float], **kwargs) -> float:

        pass

    @staticmethod
    def is_float(value: str) -> bool:

        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(value: str) -> bool:

        try:
            int(value)
            return True
        except ValueError:
            return False


class PIDControlAlgorithm(ControlAlgorithmBase):

    def __init__(
            self,
            Kp: float = 1.0,
            Ki: float = 0,
            Kd: float = 0,
            set_point: float = 100.0,
            integral_limit: float = 5.0
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point

        self.previous_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit
        logging.info(
            f"PIDControlAlgorithm initialized with "
            f"Kp={Kp}, Ki={Ki}, Kd={Kd}, set_point={set_point}, integral_limit={integral_limit}"
        )

    def calculate_extrusion(self, inference_results: List[float], **kwargs) -> int:
        try:
            # First log all current inference_results values
            logging.debug(f"PID: Received inference_results={inference_results}")

            if not inference_results:
                logging.warning("No valid inference results for extrusion_output calculation.")
                return 100

            inference = np.array(inference_results) * 50 + 100

            Boxplot = kwargs.get('Boxplot', 1)

            if Boxplot == 1:
                model = ExponentialSmoothing(inference, trend='add').fit()
                pred = model.fittedvalues
                std_resid = np.std(model.resid)
                mask_es = np.abs(inference - pred) > 3 * std_resid
                filtered_inference = inference[~mask_es]
            else:
                filtered_inference = inference

            error = self.set_point - np.mean(filtered_inference)

            logging.debug(f"PID: output_ob={filtered_inference}, Set Point={self.set_point}, Error={error}")

            # Calculate integral and derivative
            self.integral += error
            # Integral anti-windup
            self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
            derivative = error - self.previous_error
            logging.debug(f"PID: Integral={self.integral}, Derivative={derivative}")

            # Calculate control output
            extrusion_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
            logging.debug(
                f"PID: Kp*Error={self.Kp * error}, "
                f"Ki*Integral={self.Ki * self.integral}, "
                f"Kd*Derivative={self.Kd * derivative}, "
                f"Extrusion Output={extrusion_output}"
            )

            self.previous_error = error

            output_p = kwargs.get('output_p', 100)

            logging.info(f"PID: Calculated increment={extrusion_output}")
            logging.info(f"PID:     Output={extrusion_output + output_p}")
            return int(extrusion_output + output_p)
        except Exception as e:
            logging.error(f"Error calculating extrusion_output with PID control algorithm: {e}", exc_info=True)
            return 100


class FuzzyControlAlgorithm(ControlAlgorithmBase):


    def __init__(
            self,
            set_point: int = 100
    ):
        if not isinstance(set_point, int):
            raise ValueError(f"set_point must be integer type, but received {type(set_point)}")
        self.set_point = set_point
        self.last_output = 0
        self.last_delta = 0
        self.last_control_value = 0

        logging.info(
            f"FuzzyControlAlgorithm initialized with set_point={set_point}"
        )

        cv_value = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cv_value')

        delta = ctrl.Antecedent(np.arange(-50, 51, 1), 'delta')

        error_sign = ctrl.Antecedent(np.arange(-1, 2, 1), 'error_sign')

        control_output = ctrl.Consequent(np.arange(-50, 51, 1), 'control_output')

        cv_value['low'] = fuzz.gaussmf(cv_value.universe, , )   # Low coefficient of variation
        cv_value['middle'] = fuzz.gaussmf(cv_value.universe, , )   # Medium coefficient
        cv_value['high'] = fuzz.gaussmf(cv_value.universe, , )   # High coefficient

        error_sign['posi'] = fuzz.gaussmf(error_sign.universe, , )   # Positive
        error_sign['zero'] = fuzz.gaussmf(error_sign.universe, , )   # Zero
        error_sign['nega'] = fuzz.gaussmf(error_sign.universe, , )  # Negative

        delta['negative_large'] = fuzz.gaussmf(delta.universe, -50, 5)  # Large negative error
        delta['negative_medium'] = fuzz.gaussmf(delta.universe, -35, 5)  # Medium negative error
        delta['negative_small'] = fuzz.gaussmf(delta.universe, -20, 5)  # Small negative error
        delta['zero'] = fuzz.gaussmf(delta.universe, 0, 5)  # Zero error
        delta['positive_small'] = fuzz.gaussmf(delta.universe, 20, 5)  # Small positive error
        delta['positive_medium'] = fuzz.gaussmf(delta.universe, 35, 5)  # Medium positive error
        delta['positive_large'] = fuzz.gaussmf(delta.universe, 50, 5)  # Large positive error

        control_output['decrease_large'] = fuzz.gaussmf(control_output.universe, -50, 5)
        control_output['decrease_medium'] = fuzz.gaussmf(control_output.universe, -35, 5)
        control_output['decrease_small'] = fuzz.gaussmf(control_output.universe, -20, 5)
        control_output['decrease_nano'] = fuzz.gaussmf(control_output.universe, -5, 3)
        control_output['maintain'] = fuzz.gaussmf(control_output.universe, 0, 3)
        control_output['increase_nano'] = fuzz.gaussmf(control_output.universe, 5, 3)
        control_output['increase_small'] = fuzz.gaussmf(control_output.universe, 20, 5)
        control_output['increase_medium'] = fuzz.gaussmf(control_output.universe, 35, 5)
        control_output['increase_large'] = fuzz.gaussmf(control_output.universe, 50, 5)

        rule1 = ctrl.Rule(cv_value['high'], control_output['maintain'])
        rule9 = ctrl.Rule(cv_value['middle'] & error_sign['posi'], control_output['decrease_nano'])
        rule10 = ctrl.Rule(cv_value['middle'] & error_sign['nega'], control_output['increase_nano'])
        rule11 = ctrl.Rule(cv_value['middle'] & error_sign['zero'], control_output['maintain'])
        rule2 = ctrl.Rule(cv_value['low'] & delta['negative_large'], control_output['increase_large'])
        rule3 = ctrl.Rule(cv_value['low'] & delta['negative_medium'], control_output['increase_medium'])
        rule4 = ctrl.Rule(cv_value['low'] & delta['negative_small'], control_output['increase_small'])
        rule5 = ctrl.Rule(cv_value['low'] & delta['zero'], control_output['maintain'])
        rule6 = ctrl.Rule(cv_value['low'] & delta['positive_small'], control_output['decrease_small'])
        rule7 = ctrl.Rule(cv_value['low'] & delta['positive_medium'], control_output['decrease_medium'])
        rule8 = ctrl.Rule(cv_value['low'] & delta['positive_large'], control_output['decrease_large'])

        self.fuzzy_control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4,
                                                        rule5, rule6, rule7, rule8, rule9, rule10, rule11])
        self.fuzzy_simulation = ctrl.ControlSystemSimulation(self.fuzzy_control_system)

    def calculate_extrusion(self, inference_results: List[float], **kwargs) -> int:
        try:
            # Log inference_results
            logging.debug(f"Fuzzy: Received inference_results={inference_results}")

            if not inference_results:
                logging.warning("No valid inference results for extrusion_output calculation.")
                return 0


            inference = np.array(inference_results) * 50 + 100

            Boxplot = kwargs.get('Boxplot', 1)

            if Boxplot == 1:
                model = ExponentialSmoothing(inference, trend='add').fit()
                pred = model.fittedvalues
                std_resid = np.std(model.resid)
                mask_es = np.abs(inference - pred) > 3 * std_resid
                filtered_inference = inference[~mask_es]
            else:
                filtered_inference = inference

            mean = np.mean(filtered_inference)
            std_dev = np.std(filtered_inference)
            cv = std_dev / mean if mean != 0 else 0
            cv = np.clip(cv, 0, 1)

            delta = np.mean(filtered_inference[-10:]) - self.set_point
            delta = np.clip(delta, -50, 50)

            sign = np.sign(delta) if delta != 0 else 0

            logging.debug(
                f"Fuzzy: input={inference_results}, dalta={delta}, Set_Point={self.set_point}, CV={cv}, sign={sign}")

            self.fuzzy_simulation.input['delta'] = delta
            self.fuzzy_simulation.input['cv_value'] = cv
            self.fuzzy_simulation.input['error_sign'] = int(sign)

            self.fuzzy_simulation.compute()

            output_p = kwargs.get('output_p', 100)
            try:
                output_value = self.fuzzy_simulation.output['control_output']
                if abs(self.last_control_value) >= 10:
                    output_value = 0
                    self.last_control_value = 0
                else:
                    self.last_control_value = output_value
            except KeyError:
                logging.warning("Control output not generated, using base value")
                return output_p

            output = int(output_p + output_value)

            logging.info(f"Fuzzy: Calculated extrusion_output={output}")
            return output
        except Exception as e:
            logging.error(f"Fuzzy control exception: {str(e)}", exc_info=True)
            return kwargs.get('output_p', 100)

    def _update_state(self, output_p: int):




def ControlAlgorithmFactory(algorithm_name: str, **kwargs) -> ControlAlgorithmBase:

    algorithms = {
        "PID": PIDControlAlgorithm,
        "Fuzzy": FuzzyControlAlgorithm,
    }

    algorithm_class = algorithms.get(algorithm_name)
    if not algorithm_class:
        logging.warning(f"Unknown control algorithm '{algorithm_name}'. Using PID as default.")
        algorithm_class = PIDControlAlgorithm

    return algorithm_class(**kwargs)
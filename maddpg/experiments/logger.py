from logging import Logger
import numpy as np
import time, datetime

class Logger():
    def __init__(self, save_dir):
        self.save_log = save_dir + "/%s" %(datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
        self.action_spaces = {0 : "Raise hand", 1 : "Give away password", 2 : "Try to disarm the criminal"}
        self.neu_action_spaces = {0 : "Raise hand", 1 : "Run away", 2 : "Try to disarm the criminal"}
        self.adv_action_spaces = {0 : "Use gun to point at employee", 1 : "Use gun to point at customer", 2 : "Shoot at ceiling", 
        3 : "Ask for password", 4 : "Kill employee", 5 : "Kill customer", 6 : "Run away", 7 : "Hit employee", 8 : "Hit customer"}

    def log_action_n(self, action_n, neu_state):
        with open(self.save_log, "a") as f:
            f.write(
                "The criminal takes action %s.\n" %(self.adv_action_spaces[np.argmax(action_n[0])])
            )
            if any(neu_state):
                pass
            else:
                f.write(
                    "The customer takes action %s.\n" %(self.neu_action_spaces[np.argmax(action_n[1])])
                )
            f.write(
                "The bank employee takes action %s.\n" %(self.action_spaces[np.argmax(action_n[2])])
            )
            
    def log_episode_end_info(self, info):
        with open(self.save_log, "a") as f:
            f.write(
                info + "\n"
            )
            f.write("======================================\n")

    def log_epsiode_start_info(self):
        with open(self.save_log, "a") as f:
            f.write("New episode begins:\n")
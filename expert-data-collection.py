import gym
import csv
import pygame
from gym.utils.play import play
import numpy as np
import json
from PIL import Image
import os

env = gym.make("CarRacing-v2", render_mode="rgb_array")
default_action = np.array([0, 0, 0])
mapping = {
    (pygame.K_LEFT,): np.array([-1, 0, 0]),
    (pygame.K_UP,): np.array([0, 1, 0]),
    (pygame.K_RIGHT,): np.array([1, 0, 0]),
    (pygame.K_DOWN,): np.array([0, 0, 0.2]),
}


timestep = 0
frame_count = 0
first_time = True
directory = "expert-data"


def saveImage(data):
    path = os.path.join(directory, f"image_{timestep}.png")
    im = Image.fromarray(data)
    im.save(path)


def writeToCVS(image, action, reward):
    path = os.path.join(directory, f"data.csv")
    with open(path, "a") as file:
        writer = csv.writer(file)
        writer.writerow([image, action, reward])


def createDirectoryIfNeeded():
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)


def callback(prev_obs, obs, action, reward, terminated, truncated, info):
    global first_time, timestep, frame_count
    if frame_count > 50:
        createDirectoryIfNeeded()
        if first_time:
            writeToCVS("image", "action", "reward")
            first_time = False
        saveImage(obs)
        writeToCVS(f"image_{timestep}.png", json.dumps(action.tolist()), reward)

        timestep += 1

    frame_count += 1
    if truncated or terminated:
        frame_count = 0


play(env, keys_to_action=mapping, noop=default_action, callback=callback)

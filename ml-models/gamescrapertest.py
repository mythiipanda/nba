from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

chrome_options = Options()
service = Service('chromedriver.exe')
driver = webdriver.Chrome(service=service, options=chrome_options)

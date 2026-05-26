from brain_regions.language_area import LanguageArea
import time

print("Testing Language Area...")
language_area = LanguageArea(model_name="llama3", poll_interval=2.0)
language_area.start()

# Simulate a few states
language_area.update_state(100.0, 0.0, 10, False)
time.sleep(5)
language_area.update_state(50.0, 0.8, 100, False)
time.sleep(5)
language_area.update_state(0.0, 0.0, 200, True)
time.sleep(5)
print("Done testing.")

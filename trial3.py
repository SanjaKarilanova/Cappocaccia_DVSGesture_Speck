import samna
deviceInfos = samna.device.get_unopened_devices()
# Select the device you want to open, here we want to open the first one
print(deviceInfos)
device = samna.device.open_device(deviceInfos[0])

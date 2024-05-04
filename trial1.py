from sinabs.backend.dynapcnn.io import get_device_map
from typing import Dict

import samna
devices = samna.device.get_unopened_devices()

device_map: Dict[str, 'DeviceInfo'] = get_device_map()
print(device_map)


k=0
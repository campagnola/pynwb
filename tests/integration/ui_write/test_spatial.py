import unittest2 as unittest
import pynwb
import numpy as np
from datetime import datetime


from pynwb import NWBHDF5IO, NWBFile

nwbfile = NWBFile('the PyNWB tutorial', 'my first synthetic recording', 'EXAMPLE_ID', datetime.now(),
                  experimenter='Dr. Bilbo Baggins',
                  lab='Bag End Laboratory',
                  institution='University of Middle Earth at the Shire',
                  experiment_description='I went on an adventure with thirteen dwarves to reclaim vast treasures.',
                  session_id='LONELYMTN')

image_series = pynwb.image.ImageSeries(name="test_images",
                                       source="",
                                       data=np.random.normal(size=(10,100,100)),
                                       starting_time=0.0,
                                       rate=30.0)

nwbfile.add_acquisition(image_series)

io = NWBHDF5IO('spatial_test.nwb', 'w')
io.write(nwbfile)
io.close()


io = NWBHDF5IO('spatial_test.nwb', 'r')
nwbfile = io.read()

image = nwbfile.get_acquisition("test_images")
import unittest
import tempfile
import os
from datetime import datetime
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.icephys import CurrentClampSeries, CurrentClampStimulusSeries, IntracellularElectrode


class TestHDF5Roundtrip(unittest.TestCase):
    def setUp(self):
        fh, self.filename = tempfile.mkstemp()
        os.close(fh)

    def tearDown(self):
        os.remove(self.filename)

    def test_roundtrip(self):
        """Test writing NWB to HDF5, then reading back again.
        """
        start_time = datetime(2017, 4, 3, 11, 0, 0)
        create_date = datetime(2017, 4, 15, 12, 0, 0)

        nwbfile = NWBFile(
            source='roundtrip integration test', 
            session_description='', 
            identifier='', 
            session_start_time=start_time, 
            file_create_date=create_date
        )

        dur = 1.0
        srate = 50000.
        npts = int(srate*dur)
        cmd = np.zeros(npts)
        pulse = (0.2, 0.6)
        cmd[int(pulse[0] * srate):int(pulse[1] * srate)] = 200e-12
        data = np.random.normal(size=npts, scale=400e-6, loc=-70e-3)
        data += cmd * 10e6

        elec = IntracellularElectrode(
            name="elec0", 
            source='',
            slice='',
            resistance='',
            seal='', 
            description='',
            location='',
            filtering='',
            initial_access_resistance='',
            device=''
        )

        ccas = CurrentClampSeries(
            name='ccas',
            source='primary',
            data=data,
            unit='V',
            starting_time=123.6,
            rate=srate,
            electrode=elec,
            gain=400e12,
            bias_current=0.,
            bridge_balance=0.,
            capacitance_compensation=0.
        )

        nwbfile.add_acquisition(ccas)

        ccss = CurrentClampStimulusSeries(
            name="ccss", 
            source="command", 
            data=cmd, 
            unit='A',
            starting_time=123.6,
            rate=srate,
            electrode=elec,
            gain=0.02
        )
        
        # with self.assertRaises(TypeError):
        #    nwbfile.add_acquisition(ccss)
        
        nwbfile.add_stimulus(ccss)

        io = NWBHDF5IO(self.filename, 'w')
        io.write(nwbfile)
        io.close()

        io = NWBHDF5IO(self.filename)
        nwbfile = io.read()
        ccss2 = nwbfile.get_stimulus('ccss')
        ccas2 = nwbfile.get_acquisition('ccas')

        self.assertTrue(np.all(ccss2.data == ccss.data))
        self.assertTrue(np.all(ccas2.data == ccas.data))

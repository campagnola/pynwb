"""
.. _basics:

NWB basics
==========

This example will focus on the basics of working with an :py:class:`~pynwb.file.NWBFile` object,
including writing and reading of and NWB file.

"""

####################
# The NWB file
# ------------
#
#

from datetime import datetime
from pynwb import NWBFile

start_time = datetime(2017, 4, 3, 11, 0, 0)
create_date = datetime(2017, 4, 15, 12, 0, 0)

nwbfile = NWBFile('PyNWB tutorial', 'demonstrate NWBFile basics', 'NWB123', start_time,
                  file_create_date=create_date)

####################
# .. _basic_timeseries:
#
# Time series data
# ----------------
#
# PyNWB stores time series data using the :py:class:`~pynwb.base.TimeSeries` class and its subclasses.
# The main components of a :py:class:`~pynwb.base.TimeSeries` are the *data* and the *timestamps*.
# You will also need to supply a *source* and *description* of the data and the unit for *data*.

from pynwb import TimeSeries

data = list(range(100, 200, 10))
timestamps = list(range(10))
test_ts = TimeSeries('test_timeseries', 'PyNWB tutorial', data, 'SIunit', timestamps=timestamps)

####################
# Alternatively, if your recordings are sampled at a uniform rate, you can supply *starting_time*
# and *rate*.

test_ts = TimeSeries('test_timeseries', 'PyNWB tutorial', data, 'SIunit', starting_time=0.0, rate=1.0)

####################
# Using this scheme says that this :py:class:`~pynwb.base.TimeSeries` started recording 0 seconds after
# *start_time* stored in the :py:class:`~pynwb.file.NWBFile` and sampled every second.
#
# :py:class:`~pynwb.base.TimeSeries` objects can be added directly to your :py:class:`~pynwb.file.NWBFile` using
# the methods :py:func:`~pynwb.file.NWBFile.add_acquisition`, :py:func:`~pynwb.file.NWBFile.add_stimulus`
# and :py:func:`~pynwb.file.NWBFile.add_stimulus_template`. Which method you use depends on the source of the
# data: use :py:func:`~pynwb.file.NWBFile.add_acquisition` to indicated *acquisition* data,
# :py:func:`~pynwb.file.NWBFile.add_stimulus` to indicate *stimulus* data, and
# :py:func:`~pynwb.file.NWBFile.add_stimulus_template` to store stimulus templates [#]_.

nwbfile.add_acquisition(test_ts)

####################
# .. _basic_data_interfaces:
#
# Data interfaces
# ---------------
#
# NWB provides the concept of a *data interface*--an object for a standard
# storage location of specific types of data--through the :py:class:`~pynwb.base.NWBDataInterface` class.
# For example, :py:class:`~pynwb.ecephys.LFP` provides a container for holding one or more
# :py:class:`~pynwb.ecephys.ElectricalSeries` objects that store local-field potential data. By putting
# your LFP data into an :py:class:`~pynwb.ecephys.LFP` container,  downstream users and tools know where
# to look to retrieve LFP data. For a comprehensive list of available data interfaces, see the
# :ref:`overview page <modules_overview>`
#
# :py:class:`~pynwb.base.NWBDataInterface` objects can be added as acquisition data, or as members
# of a :ref:`ProcessingModule <basic_procmod>`
#
# For the purposes of demonstration, we will use a :py:class:`~pynwb.ecephys.LFP` data interface.

from pynwb.ecephys import LFP

lfp = LFP('PyNWB tutorial')
nwbfile.add_acquisition(lfp)

####################
# Each data interface stores its own type of data. We suggest you read the documentation for the
# data interface of interest in the :ref:`API documentation <api_docs>` to figure out what data the
# data interface allows and/or requires and what methods you will need to call to add this data.

####################
# .. _basic_procmod:
#
# Processing modules
# ------------------
#
# *Processing modules* are used for storing a set of data interfaces that are related to a particular
# processing workflow. For example, if you want to store intermediate and final results of a spike sorting workflow,
# you would create a :py:class:`~pynwb.base.ProcessingModule` that contains data interfaces that represent
# the common steps in spike sorting e.g. :py:class:`~pynwb.ecephys.EventDetection`,
# :py:class:`~pynwb.ecephys.EventWaveform`,  :py:class:`~pynwb.ecephys.FeatureExtraction`,
# :py:class:`~pynwb.ecephys.Clustering`, :py:class:`~pynwb.ecephys.ClusterWaveform`.
#
# Processing modules can be created using :py:func:`~pynwb.file.NWBFile.create_processing_module`:

created_mod = nwbfile.create_processing_module('created_mod', 'PyNWB tutorial', 'example module')

####################
# or by directly calling the constructor and adding to the :py:class:`~pynwb.file.NWBFile` using
# :py:func:`~pynwb.file.NWBFile.add_processing_module`:

from pynwb import ProcessingModule

added_mod = ProcessingModule('added_mod', 'PyNWB tutorial', 'example module')
nwbfile.add_processing_module(added_mod)

####################
# You can add data to your processing module using the method
# :py:func:`~pynwb.base.ProcessingModule.add_data_interface`.
# Lets make another :py:class:`~pynwb.base.TimeSeries` and then add it to the
# :py:class:`~pynwb.base.ProcessingModule` we just added.

data = list(range(0, 100, 10))
timestamps = list(range(10))
mod_ts = TimeSeries('ts_for_mod', 'PyNWB tutorial', data, 'SIunit', timestamps=timestamps)
added_mod.add_data_interface(mod_ts)

####################
# .. _basic_epochs:
#
# Epochs
# ------
#
# Epochs can be added to an NWB file using the method :py:func:`~pynwb.file.NWBFile.create_epoch`.
# The first argument is a description of the epoch, the second and third argument are the start time
# and stop time, respectively. The fourth argument is one or more tags for labelling the epoch,
# and the fifth argument is a list of all the :py:class:`~pynwb.base.TimeSeries` that the epoch applies
# to.

nwbfile.create_epoch('the first epoch', 2.0, 4.0, ['first', 'example'], [test_ts, mod_ts])
nwbfile.create_epoch('the second epoch', 6.0, 8.0, ['second', 'example'], [test_ts, mod_ts])

####################
# .. _basic_writing:
#
# Writing an NWB file
# -------------------
#
# NWB I/O is carried out using the :py:class:`~pynwb.NWBHDF5IO` class [#]_. This class is responsible
# for mapping an :py:class:`~pynwb.file.NWBFile` object into HDF5 according to the NWB schema.
#
# To write an :py:class:`~pynwb.file.NWBFile`, use the :py:func:`~pynwb.form.backends.io.FORMIO.write` method.

from pynwb import NWBHDF5IO

io = NWBHDF5IO('basic_example.nwb', 'w')
io.write(nwbfile)
io.close()

####################
# You can also use :py:func:`~pynwb.NWBHDF5IO` as a context manager:

with NWBHDF5IO('basic_example.nwb', 'w') as io:
    io.write(nwbfile)

####################
# .. _basic_reading:
#
# Reading an NWB file
# -------------------
#
# As with writing, reading is also carried out using the :py:class:`~pynwb.NWBHDF5IO` class.
# To read the NWB file we just wrote, using construct another :py:class:`~pynwb.NWBHDF5IO` object,
# and use the :py:func:`~pynwb.form.backends.io.FORMIO.read` method to retrieve an
# :py:class:`~pynwb.file.NWBFile` object.

io = NWBHDF5IO('basic_example.nwb')
nwbfile = io.read()

####################
# For reading, we cannot use :py:class:`~pynwb.NWBHDF5IO` as a context manager, since the resulting
# :py:class:`~pynwb.NWBHDF5IO` gets closed and deleted when the context completes [#]_.

####################
# .. _basic_retrieving_data:
#
# Retrieving data from an NWB file
# --------------------------------
#
# Most of the methods we used above to write data are paired with a getter method for getting your data back.
#
# Lets start with the :py:class:`~pynwb.base.TimeSeries` object we wrote. Above, we added it as
# acquisition data using the method :py:func:`~pynwb.file.NWBFile.add_acquisition`. We can get it
# back in a couple ways. The first we just mentioned--a simple getter method. In the case of acquisition
# data, the method is :py:func:`~pynwb.file.NWBFile.get_acquisition`. The only argument this method needs
# is the name of the object you are trying to get. We named our :py:class:`~pynwb.base.TimeSeries`
# "test_timeseries":

ts = nwbfile.get_acquisition('test_timeseries')

####################
# If you are not into *getter* methods, you can also retrieve this data by pulling it out of the
# :py:func:`~pynwb.file.NWBFile.acquisition` property. This property supports dict-like indexing. Again,
# all we need to supply is the name of the object we are looking for:

ts = nwbfile.acquisition['test_timeseries']

####################
# We can also get the :py:class:`~pynwb.ecephys.LFP` object back. When we created the :py:class:`~pynwb.ecephys.LFP`
# object, we did not supply a name, so the name defaulted to "LFP" [#]_.

lfp = nwbfile.acquisition['LFP']

####################
# Just like acquisition data, we can get processing modules back in the same manner. We created two above.
# Lets read both, but using the two different ways. The first way,
# calling :py:func:`~pynwb.file.NWBFile.get_processing_module`:

created_mod = nwbfile.get_processing_module('created_mod')

####################
# And the second way, indexing into :py:func:`~pynwb.file.NWBFile.modules`

added_mod = nwbfile.modules['added_mod']

####################
# Now that we have our :py:class:`~pynwb.base.ProcessingModule` back, we can get the :py:class:`~pynwb.base.TimeSeries`
# that we added to it back. Similar to :py:class:`~pynwb.file.NWBFile`, we have two ways of gettings this data back.
# The first is by using the getter :py:func:`~pynwb.base.ProcessingModule.get_data_interface` and passing in
# the name of the object we want back.

mod_ts = added_mod.get_data_interface('ts_for_mod')

####################
# The second way is by indexing directly into the :py:class:`~pynwb.base.ProcessingModule` object and passing
# the name of the object we want back.

mod_ts = added_mod['ts_for_mod']

####################
# .. [#] Stimulus template data may change in the near future. The NWB team will work with interested parties
#    at the `4th NWB Hackathon <hck04_>`_ to refine the schema for storing stimulus template data.
#
# .. [#] HDF5 is currently the only backend supported by NWB.
#
# .. [#] Neurodata sets can be *very* large, so individual components of the dataset are only loaded into memory when
#    you requst them. This functionality is only possible if an open file handle is kept around until users want to
#    load data.
#
# .. [#] Some data interface objects have a default name. This default name is the type of the data interface. For
#    example, the default name for :py:class:`~pynwb.ophys.ImageSegmentation` is "ImageSegmentation" and the default
#    name for :py:class:`~pynwb.ecephys.EventWaveform` is "EventWaveform".

####################
# .. _hck04: https://github.com/NeurodataWithoutBorders/nwb_hackathons/tree/master/HCK04_2018_Seattle

import time
from pypylon import pylon
import cv2

#Camera Start

class CameraPylon:
    def __init__(self,id=0, exposure_us = 30000, gain = 0.0):
        """
        Init

        * When exposure_us is zero, auto exposure is enabled.
        * When gain is zero, auto gain is enabled.
        """
        self.factory = pylon.TlFactory.GetInstance()
        # DeviceInfo class get the number of connected cameras, camera name, etc.
        self.devices = self.factory.EnumerateDevices()
        if len(self.devices) == 0:
            raise Exception('no camera present.')
        # Create a class to control the camera.
        self.camera = pylon.InstantCamera(self.factory.CreateDevice(self.devices[id]))
        # The camera settings cannot be changed without opening.
        self.camera.Open()
        if exposure_us == 0:
            self.camera.ExposureAuto.SetValue('Continuous')
        else:
            self.camera.ExposureAuto.SetValue('Off')
            self.camera.ExposureTime.SetValue(exposure_us)
        if gain == -1.0:
            self.camera.GainAuto.SetValue('Continuous')
        else:
            self.camera.GainAuto.SetValue('Off')
            self.camera.Gain.SetValue(gain)
        print('--------------------')
        print('CameraPylon.__init__')
        print('--------------------')
        print('Name = {0}'.format(self.devices[id].GetFriendlyName()))
        print('Width = {0}'.format(self.camera.Width.GetValue()))
        print('Height = {0}'.format(self.camera.Height.GetValue()))
        print('ExposureAuto = {0}'.format(self.camera.ExposureAuto.GetValue()))
        print('ExposureTime = {0}[us]'.format(self.camera.ExposureTime.GetValue()))
        print('GainAuto = {0}'.format(self.camera.GainAuto.GetValue()))
        print('Gain = {0}'.format(self.camera.Gain.GetValue()))
        self.camera.Close()
        # Set ImageFormatConverter.
        self.converter_bgr = pylon.ImageFormatConverter()
        self.converter_bgr.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter_bgr.OutputBitAlignment = 'MsbAligned'
        # Set display magnification.
        self.disp_mag = 50

    def open(self):
        """
        Open.
        """
        self.camera.Open()

    def close(self):
        """
        Close.
        """
        self.camera.Close()


    def setExposureTime(self, exposure_us=10000, en_print=True):
        """
        Set exposure time.

        * When exposure_us is zero, auto exposure is enabled.
        * When en_print is True, display the set value.
        """
        if not self.camera.IsOpen():
            raise Exception('camera is not open.')
        if exposure_us == 0:
            self.camera.ExposureAuto.SetValue('Continuous')
        else:
            self.camera.ExposureAuto.SetValue('Off')
            self.camera.ExposureTime.SetValue(exposure_us)
        if en_print:
            print('ExposureAuto = {0}'.format(self.camera.ExposureAuto.GetValue()))
            print('ExposureTime = {0}[us]'.format(self.camera.ExposureTime.GetValue()))


    def setGain(self, gain=0.0, en_print=True):
        """
        Set gain.

        * When gain is -1, auto gain is enabled.
        * When en_print is True, display the set value.
        """
        if not self.camera.IsOpen():
            raise Exception('camera is not open.')
        if gain == -1.0:
            self.camera.GainAuto.SetValue('Continuous')
        else:
            self.camera.GainAuto.SetValue('Off')
            self.camera.Gain.SetValue(gain)
        if en_print:
            print('GainAuto = {0}'.format(self.camera.GainAuto.GetValue()))
            print('Gain = {0}[dB]'.format(self.camera.Gain.GetValue()))


    def grab(self, timeout=1000, en_print=True):
        """
        Grab.

        * Run StartGrabbing and StopGrabbing each time.
        * All convert to 24-bit BGR.
        * When en_print is True, display the set value.
        """
        if not self.camera.IsOpen():
            raise Exception('camera is not open.')
        t_start = time.time()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        grabResult = self.camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
        rslt_conv = self.converter_bgr.Convert(grabResult)
        grabResult.Release()
        self.camera.StopGrabbing()
        proc_time = time.time() - t_start
        if en_print:
            print('grab time : {0} ms'.format(proc_time))
        return rslt_conv.GetArray()

    def view(self):

        pass


    def _changeExposure(self, val):
        exp = cv2.getTrackbarPos("Exp[ms]", "img")
        self.setExposureTime(exp * 1000, False)

    def _changeGain(self, val):
        gain = cv2.getTrackbarPos("Gain[dB]", "img")
        self.setGain(gain, False)

    def _changeMag(self, val):
        mag = cv2.getTrackbarPos("Mag[%]", "img")
        self.disp_mag = int(mag)

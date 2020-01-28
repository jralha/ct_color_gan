#%%
import cv2
import glob
import numpy as np
from tqdm.auto import tqdm
import sys
import vtk

#%%
folder = 'results\\Libra_CX1_T1_221_184_1485_Energy1\\ct_color\\bg_fix'
files = glob.glob(folder+'\\*.png')

files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

filePath = vtk.vtkStringArray()
filePath.SetNumberOfValues(len(files))
for i in range(0,len(files),1):
    filePath.SetValue(i,files[i])

reader=vtk.vtkPNGReader()
reader.SetFileNames(filePath)
reader.SetDataSpacing(1,1,1)
reader.Update()

#%%
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.SetColorSpaceToRGB()

rgbConverter = vtk.vtkImageMapToColors()
rgbConverter.SetOutputFormatToRGB()
rgbConverter.SetLookupTable(vtk.vtkScalarsToColors())



opacity = vtk.vtkPiecewiseFunction()
opacity.AddPoint(0, 0)
opacity.AddPoint(250, 1)
# opacity.AddPoint(100, 0.8)
# opacity.AddPoint(200, 1)


volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetGradientOpacity(opacity)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.SetIndependentComponents(0)
# volumeProperty.ShadeOn()


# volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper() 
volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()

volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetBlendModeToComposite()

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)



ren = vtk.vtkRenderer()
ren.AddVolume(volume)
#No need to set by default it is black
ren.SetBackground(0, 0, 0)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(900, 900)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

interactor.Initialize()
renWin.Render()
interactor.Start()

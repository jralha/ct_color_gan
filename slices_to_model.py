#%%
import cv2
import glob
import numpy as np
from tqdm.auto import tqdm
import sys
import vtk

#%%
# folder = 'results\\ct_color\\test_latest\\bg_fix'
folder = 'temp_data\\Libra_CX1_T1_221_184_1485_Energy1'
files = glob.glob(folder+'\\*.png')

files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

print(files)

filePath = vtk.vtkStringArray()
filePath.SetNumberOfValues(len(files))
for i in range(0,len(files),1):
    filePath.SetValue(i,files[i])

reader=vtk.vtkPNGReader()
reader.SetFileNames(filePath)
reader.SetDataSpacing(1,1,1)
reader.Update()

#%%
volumeGradientOpacity = vtk.vtkPiecewiseFunction()
volumeGradientOpacity.AddPoint(0,   0)
volumeGradientOpacity.AddPoint(300, 1)

volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetGradientOpacity(volumeGradientOpacity)
volumeProperty.IndependentComponentsOff()
volumeProperty.SetInterpolationTypeToNearest()

#To make color volumes in opengl images need to me in RGBA, not RGB
#Opengl is a lot faster, should use it ideally.
# volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper() 
volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()


volumeMapper.SetInputConnection(reader.GetOutputPort())
volumeMapper.SetBlendModeToComposite()

volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)
 
ren = vtk.vtkRenderer()
ren.AddVolume(volume)
ren.SetBackground(0, 0, 0)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(900, 900)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

interactor.Initialize()
renWin.Render()
interactor.Start()

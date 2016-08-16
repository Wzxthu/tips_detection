### Use IJK coordinates to trigger an automatic needle segmentation
## You have to set the tempolate first
import slicer
import vtk
import numpy as np
def test(path):
    # path = 'C:/Users/fff/Desktop/64-latest-black.save'
    res = np.load(path)

    ## choose remains after metric,

    # remain = [2,11,18,19,28,33,38]
    # res = res[[remain]]
    ## delete wrong samples after metric
    # delete = [2,8,9,10,11,14,18,19,20,26,28,29,30,33,35,37,38,39]
    # res = np.delete(res,delete,0)
    # start to find needles using manual segmentation

    w = slicer.modules.NeedleFinderWidget
    l = w.logic
    l.placeAxialLimitMarker()
    tips = res.tolist() # for two tips for example: [[220,100,90], [150,100,90]]
    names = ['seg'+str(i) for i in range(len(res))] # for two tips for example: ['seg1','seg2']
    script = False
    m = vtk.vtkMatrix4x4()
    volumeNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    volumeNode.GetIJKToRASMatrix(m)
    imageData = volumeNode.GetImageData()
    spacing = volumeNode.GetSpacing()
    l.needleDetectionThread(tips, imageData, spacing=spacing, script=script, names=names)



from __main__ import vtk, qt, ctk, slicer
import numpy as np 

#
# Correlation
#

class Correlation:
  def __init__(self, parent):
    parent.title = "Correlation" 
    parent.categories = ["Examples"]
    parent.dependencies = []
    parent.contributors = ["Carla Garcia Gallego", "Helena Puente Diaz", "Marta Rodrigo Sebastian"] 
    parent.helpText = """
    Se calcula la correlacion entre dos imagenes.
    """
    parent.acknowledgementText = """
    """ 
    self.parent = parent

#
# CorrelationWidget
#

class CorrelationWidget:
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()

  def setup(self):
    #create collapsible button
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "My collapsible Menu"
    #bind collapsible button to root layout
    self.layout.addWidget(collapsibleButton)

    #new layout for collapsible button
    self.formLayout = qt.QFormLayout(collapsibleButton)

    #volume selector
    self.formFrame = qt.QFrame(collapsibleButton)

    #set the layout to horizontal
    self.formFrame.setLayout(qt.QHBoxLayout())

    #bind new frame to existing layout in collapsible menu
    self.formLayout.addWidget(self.formFrame)

    #create new volume selector (INPUT 1)
    self.inputSelector1 = qt.QLabel("Input volume 1: ", self.formFrame)
    #bind the selector to your frame
    self.formFrame.layout().addWidget(self.inputSelector1)
    
    self.inputSelector1 = slicer.qMRMLNodeComboBox(self.formFrame)
    self.inputSelector1.nodeTypes = (("vtkMRMLScalarVolumeNode"),"")
    self.inputSelector1.addEnabled = False
    self.inputSelector1.removeEnabled = False
    #bind the current volume selector to the current scene of slicer
    self.inputSelector1.setMRMLScene(slicer.mrmlScene)
    #bind now the input selector to the frame
    self.formFrame.layout().addWidget(self.inputSelector1)

    #create new volume selector (INPUT 2)
    self.inputSelector2 = qt.QLabel("Input volume 2: ", self.formFrame)
    #bind the selector to your frame
    self.formFrame.layout().addWidget(self.inputSelector2)
    
    self.inputSelector2 = slicer.qMRMLNodeComboBox(self.formFrame)
    self.inputSelector2.nodeTypes = (("vtkMRMLScalarVolumeNode"),"")
    self.inputSelector2.addEnabled = False
    self.inputSelector2.removeEnabled = False
    #bind the current volume selector to the current scene of slicer
    self.inputSelector2.setMRMLScene(slicer.mrmlScene)
    #bind now the input selector to the frame
    self.formFrame.layout().addWidget(self.inputSelector2)

    #Boton que devuelve la correlacion entre dos imagenes 
    button = qt.QPushButton("Get Correlation")
    button.toolTip = "Devuelve la correlacion entre dos imagenes"
    button.connect("clicked(bool)", self.informationButtonClicked)
    #bind button to frame
    self.formFrame.layout().addWidget(button)

    #Rectangulo donde va a aparecer la correlacion
    self.textfield = qt.QTextEdit()
    self.textfield.setReadOnly(True)
    #bind textfield to frame
    self.formFrame.layout().addWidget(self.textfield)

  def informationButtonClicked(self): 
    inputVolume1 = self.inputSelector1.currentNode()
    inputVolume2 = self.inputSelector2.currentNode()

    #Matriz de pixel de cada input
    matriz1 = slicer.util.arrayFromVolume(inputVolume1)
    matriz2 = slicer.util.arrayFromVolume(inputVolume2)

    #Dimensiones de las matrices
    matriz1_dim = matriz1.shape
    matriz2_dim = matriz2.shape

    min_x = min(matriz1_dim[0], matriz2_dim[0])
    min_y = min(matriz1_dim[1], matriz2_dim[1])
    min_z = min(matriz1_dim[2], matriz2_dim[2])

    #Resize de las matrices para que sean iguales
    matriz1_redim = np.resize(matriz1, (min_x, min_y, min_z))
    matriz2_redim = np.resize(matriz2, (min_x, min_y, min_z))

    #Correlacion
    n = min_x*min_y*min_z

    des_m1 = np.std(matriz1_redim)
    des_m2 = np.std(matriz2_redim)

    media_m1 = np.mean(matriz1_redim)
    media_m2 = np.mean(matriz2_redim)

    cor = (np.sum((1/(des_m1*des_m2))*(matriz1_redim-media_m1)*(matriz2_redim-media_m2)))/n

    self.textfield.setText(cor)
    #self.textfield.insertPlainText(cor)
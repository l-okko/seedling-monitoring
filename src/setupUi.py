# I was tiered of looking at the images by saving them, so I decided to create a GUI to display the images and also to count the plants in the image.
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QFileDialog

import plant_detection as pd 
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from config import threshold_h_lower, threshold_h_upper, threshold_s_lower, threshold_s_upper, threshold_v_lower, threshold_v_upper, errosion_iterations_default, dilation_iterations_default
from config import CB_SIZE, CB_MEASUREMENTS, CORNER_OFFSET, FRAME_MEASUREMENTS

import image_correction
import scales
import file_handler
import testing_linus
import visualizer


class Ui_MainWindow(object):
    calibration_path = "cal.jpg"
    path = None
    perspective_transform = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1318, 925)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(parent=self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1302, 868))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(parent=self.horizontalLayoutWidget)
        self.label.setMinimumSize(QtCore.QSize(720, 480))
        self.label.setMaximumSize(QtCore.QSize(1280, 720))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(".\\seedling2.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        
        # Dilation and erosion spin boxes
        self.dilation_iterations_label = QtWidgets.QLabel(parent=self.horizontalLayoutWidget)
        self.dilation_iterations_label.setObjectName("dilation_iterations_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.dilation_iterations_label)
        self.dilation_iterations_spinBox = QtWidgets.QSpinBox(parent=self.horizontalLayoutWidget)
        self.dilation_iterations_spinBox.setObjectName("dilation_iterations_spinBox")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.dilation_iterations_spinBox)
        self.errosion_iterations_label = QtWidgets.QLabel(parent=self.horizontalLayoutWidget)
        self.errosion_iterations_label.setObjectName("errosion_iterations_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.errosion_iterations_label)
        self.errosion_iterations_spinBox = QtWidgets.QSpinBox(parent=self.horizontalLayoutWidget)
        self.errosion_iterations_spinBox.setObjectName("errosion_iterations_spinBox")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.errosion_iterations_spinBox)
        self.verticalLayout_2.addLayout(self.formLayout)
        
        # Count plants button
        self.count_plants_button = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.count_plants_button.setObjectName("count_plants_button")
        self.verticalLayout_2.addWidget(self.count_plants_button)
        self.count_plants_button.clicked.connect(self.count_plants)
        self.verticalLayout_2.addWidget(self.count_plants_button)

        self.open_file_dialog_button = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.open_file_dialog_button.setObjectName("open_file_dialog_button")
        self.open_file_dialog_button.clicked.connect(self.open_file_dialog)
        self.verticalLayout_2.addWidget(self.open_file_dialog_button)
        
        # parameters Search button
        self.get_params_search_button = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.get_params_search_button.setObjectName("get_params_search_button")
        self.get_params_search_button.clicked.connect(self.test_params)
        self.verticalLayout_2.addWidget(self.get_params_search_button)

        
        self.calibrate_button = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.calibrate_button.setObjectName("calibrate_button")
        self.verticalLayout_2.addWidget(self.calibrate_button)
        self.calibrate_button.clicked.connect(self.open_calibration_file_dialog)

        # add a remove calibration button
        self.remove_calibration_button = QtWidgets.QPushButton(parent=self.horizontalLayoutWidget)
        self.remove_calibration_button.setObjectName("remove_calibration_button")
        self.verticalLayout_2.addWidget(self.remove_calibration_button)
        self.remove_calibration_button.clicked.connect(self.remove_calibration)


        self.file_label = QtWidgets.QLabel(parent=self.horizontalLayoutWidget)
        self.file_label.setObjectName("file_label")
        self.verticalLayout_2.addWidget(self.file_label)
        self.textEdit = QtWidgets.QTextEdit(parent=self.horizontalLayoutWidget)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout.addLayout(self.verticalLayout)

        # add a plain text edit to the layout called hsv_params_textEdit
        self.hsv_params_textEdit = QtWidgets.QTextEdit(parent=self.horizontalLayoutWidget)
        self.hsv_params_textEdit.setObjectName("hsv_params_textEdit")
        self.verticalLayout_2.addWidget(self.hsv_params_textEdit)
        self.horizontalLayout.addLayout(self.verticalLayout)
        


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1318, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.dilation_iterations_label.setText(_translate("MainWindow", "Dilation Iterations"))
        # set a default value for the spin boxes
        self.dilation_iterations_spinBox.setValue(dilation_iterations_default)
        self.errosion_iterations_spinBox.setValue(errosion_iterations_default)
        self.errosion_iterations_label.setText(_translate("MainWindow", "Erositon Iteratins"))
        self.count_plants_button.setText(_translate("MainWindow", "Find Plants"))
        self.open_file_dialog_button.setText(_translate("MainWindow", "Open File"))
        self.get_params_search_button.setText(_translate("MainWindow", "Dilation/Erosion Test"))
        
        # write the hsv parameters to the hsv_params_textEdit # if I pot the text like this it looks better
        # set the text in the hsv_params_textEdit only 2 decimal places
        self.hsv_params_textEdit.setText(f"""
Threshold H lower: {threshold_h_lower:.2f}\n
Threshold H upper: {threshold_h_upper:.2f}\n
Threshold S lower: {threshold_s_lower:.2f}\n
Threshold S upper: {threshold_s_upper:.2f}\n
Threshold V lower: {threshold_v_lower:.2f}\n
Threshold V upper: {threshold_v_upper:.2f}""")
        
        self.calibrate_button.setText(_translate("MainWindow", "Calibrate"))
        self.file_label.setText(_translate("MainWindow", "file"))
        self.remove_calibration_button.setText(_translate("MainWindow", "Remove Calibration"))


    def remove_calibration(self):
        self.perspective_transform = None
        self.textEdit.setText("Calibration removed")
        self.file_label.setText("file")
        # update the ui
        QtWidgets.QApplication.processEvents()

    def test_params(self):
        self.textEdit.clear()
        self.textEdit.setText("Testing parameters...")
        self.textEdit.append("This might take a while")
        # update the ui
        QtWidgets.QApplication.processEvents()

        # check if an image was loaded
        if self.file_label.text() == "file":
            self.textEdit.setText("Please load an image first")
            return
        
        # load the image
        if self.perspective_transform is None:
            img = plt.imread(self.file_label.text())
            self.textEdit.append("No calibration set, using the original image")
        else:
            img = plt.imread("tmp/corrected_img.jpg")

        fig, minima = pd.create_errosion_dilation_search(img)
        fig.savefig("test_params.png", dpi = 250)
        self.label.setPixmap(QtGui.QPixmap("test_params.png"))

        self.textEdit.append("Suggestions:")
        # print the minima
        for i in range(len(minima)):
            self.textEdit.append(f"Errosions: {minima[i][0]}, Dilation: {minima[i][1]} -> {minima[i][2]} plants\n")
        
        print(minima)


    def open_file_dialog(self):

        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)")
        if fileName:
            print(fileName)
            self.path = fileName
            self.file_label.setText(fileName)
            
            if self.perspective_transform is None:            
                self.label.setPixmap(QtGui.QPixmap(fileName))
                self.label.setScaledContents(True)
                self.label.adjustSize()
                self.label.show()
            else:
                image, _ = file_handler.load_image(fileName)
                corrected_img = image_correction.transform_image(image, self.perspective_transform)

                # image color to RGB
                corrected_img = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)

                # check if a tmp folder exists
                if not os.path.exists("tmp"):
                    os.makedirs("tmp")

                # save the corrected image
                plt.imsave("tmp/corrected_img.jpg", corrected_img)

                # display the corrected image
                self.display_image("tmp/corrected_img.jpg")



    def open_calibration_file_dialog(self):

        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)")

        if fileName:
            calibration_path = fileName
            print("Calibrating with: ", calibration_path)

            cal_image,_ = file_handler.load_image(calibration_path)

            self.perspective_transform = testing_linus.get_calibration_data(cal_image)
            # apply the perspective transform to the image
            
            # check if an image path is set
            if self.file_label.text() == "file":
                self.textEdit.append("Calibration successful")
                self.textEdit.append("Displaying the calibration image...")
                self.textEdit.setText("Please load an image to apply the transform to")
                # display the image
                self.display_image(calibration_path)

                return
            image, _ = file_handler.load_image(self.path)
            corrected_img = image_correction.transform_image(image, self.perspective_transform)


            # check if a tmp folder exists
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            
            # save the corrected image
            plt.imsave("tmp/corrected_img.jpg", corrected_img)

            self.display_image("tmp/corrected_img.jpg")



    def display_image(self, path):
        self.label.setPixmap(QtGui.QPixmap(path))
        self.label.setScaledContents(True)
        self.label.adjustSize()
        self.label.show()
        
        # update the ui
        QtWidgets.QApplication.processEvents()


    def count_plants(self):
        self.textEdit.setText("Finding plants...")
        
        # grab the parameters from the spin boxes
        dilation_iterations = self.dilation_iterations_spinBox.value()
        erosion_iterations = self.errosion_iterations_spinBox.value()

        # check if an image was loaded
        if self.path is None:
            self.textEdit.setText("Please load an image first")
            return
        
        # check if there is a calibration set and a callibrated image in the tmp folder
        if self.perspective_transform is None:
            img = plt.imread(self.file_label.text())
            self.textEdit.append("No calibration set, using the original image")
        else:
            img = plt.imread("tmp/corrected_img.jpg")

        # get the plants
        plants = pd.get_plants_from_rgb(img, None, erosion_iterations, dilation_iterations)

    
        # create a tmp folder if it does not exist
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

        visualizer.save_plant_img(img, plants, "tmp/plantHiglighted.png")

        # display the image
        self.label.setPixmap(QtGui.QPixmap("tmp/plantHiglighted.png"))
        self.label.setScaledContents(True)
        self.label.adjustSize()

        self.textEdit.append(f"Found {len(plants)} plants")
        
        # update the ui
        QtWidgets.QApplication.processEvents()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
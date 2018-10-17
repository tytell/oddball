import os, sys
import shutil
import logging

from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy as np
import wave
import subprocess

SETTINGS_FILE = "oddball.ini"

oddballbeep = {'name': 'Oddball stimuli identifier', 'type': 'group', 'children': [
        {'name': 'Note', 'type': 'str', 'value': 'A'},
        {'name': 'Volume', 'type': 'float', 'value': 1.0, 'limits': (0., 1.0)},
        {'name': 'Duration', 'type': 'float', 'value': 0.2, 'limits': (0., 10.0), 'suffix': 'sec'}
    ]}

parameters = [
    {'name': 'Instructions image', 'type': 'str', 'value': ''},
    {'name': 'Choose instructions image...', 'type': 'action'},
    {'name': 'Instructions duration', 'type': 'float', 'value': 2, 'limits': (0, np.Inf), 'suffix': 'sec'},

    {'name': 'Cue image', 'type': 'str', 'value': ''},
    {'name': 'Choose cue image...', 'type': 'action'},

    {'name': 'Oddball probability', 'type': 'float', 'value': 0.1,
     'limits': (0, 1)},
    # {'name': 'Number of each type of oddball', 'type': 'int', 'value': 10, 'visible': False},
    {'name': 'Total number of oddballs per movie', 'type': 'int', 'value': 10},
    {'name': 'Number of movies to make', 'type': 'int', 'value': 1},
    {'name': 'Duration of cue', 'type': 'float', 'value': 0.5,
     'limits': (0, np.Inf), 'suffix': 'sec'},
    {'name': 'Duration of stimulus', 'type': 'float', 'value': 0.2,
     'limits': (0, np.Inf), 'suffix': 'sec'},
    {'name': 'Background color', 'type': 'list',
     'values': ['White', 'Match', 'Black', 'Gray'], 'value': 'White'},

    {'name': 'Standard stimuli identifier', 'type': 'group', 'children': [
        {'name': 'Note', 'type': 'str', 'value': 'A'},
        {'name': 'Volume', 'type': 'float', 'value': 0.3, 'limits': (0., 1.0)},
        {'name': 'Duration', 'type': 'float', 'value': 0.1, 'limits': (0., 10.0), 'suffix': 'sec'}
    ]},
    {'name': 'Oddball stimuli identifier', 'type': 'group', 'children': [
        {'name': 'Note', 'type': 'str', 'value': 'A'},
        {'name': 'Volume', 'type': 'float', 'value': 1.0, 'limits': (0., 1.0)},
        {'name': 'Duration', 'type': 'float', 'value': 0.2, 'limits': (0., 10.0), 'suffix': 'sec'}
    ]},

    {'name': 'Internals', 'type': 'group', 'children': [
        {'name': 'Image width', 'type': 'int', 'value': 768, 'siPrefix': False, 'suffix': 'pix'},
        {'name': 'Image height', 'type': 'int', 'value': 432, 'siPrefix': False, 'suffix': 'pix'},
        {'name': 'Frame rate', 'type': 'float', 'value': 15.0, 'siPrefix': False, 'suffix': 'Hz'},
        {'name': 'Audio rate', 'type': 'float', 'value': 44100, 'siPrefix': False, 'suffix': 'Hz'},

        {'name': 'Frame name', 'type': 'str', 'value': 'frame_%04d.jpg'},
        {'name': 'Scratch directory', 'type': 'str', 'value': ''},
        {'name': 'Choose scratch directory...', 'type': 'action'}
    ]}
]

NoteFrequencies = {'A': 440, 'B': 494, 'C': 523, 'D': 587, 'E': 659, 'F': 698, 'G': 784}

class OddballWizard(QtWidgets.QWizard):
    def __init__(self, parent=None, parameters=None):
        super(OddballWizard, self).__init__(parent)

        self.oddballFiles = None
        self.oddballTypes = None
        self.noddballTypes = None
        self.standardFiles = None
        self.params = None

        self._readSettings()

        self.oddballPage = ChooseOddballPage(parent=self)
        self.standardPage = ChooseStandardPage(parent=self)
        self.parametersPage = ParametersPage(parent=self, parameters=parameters)

        self.addPage(self.oddballPage)
        self.addPage(self.standardPage)
        self.addPage(self.parametersPage)

        self.setWindowTitle('Oddball')

    def done(self, result: int):
        self._writeSettings()

        self.params = self.parametersPage.params

        self.oddballFiles = self.oddballPage.oddballFiles
        self.noddballTypes = self.oddballPage.numtypes
        self.oddballTypes = self.oddballPage.getOddballTypes()

        self.standardFiles = self.standardPage.standardFiles

        super(OddballWizard, self).done(result)

    def _readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("OddballWizard")
        self.resize(settings.value("size", type=QtCore.QSize, defaultValue=QtCore.QSize(800, 600)))
        self.move(settings.value("position", type=QtCore.QPoint, defaultValue=QtCore.QPoint(200, 200)))
        settings.endGroup()

    def _writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("SetupDialog")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.endGroup()


class ChooseOddballPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(ChooseOddballPage, self).__init__(parent)

        self.oddballFiles = None

        self.setTitle('Choose oddball images')
        self.setSubTitle('Choose the image or images that will be used for the rare ("oddball") stimulus')

        self.chooseButton = QtWidgets.QPushButton('Choose oddball image(s)...', parent=self)

        self.tableWidget = QtWidgets.QTableWidget(parent=self)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem("Filename"))
        self.tableWidget.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem("Type"))
        self.tableWidget.setColumnHidden(1, True)

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)

        self.numberWidget = QtWidgets.QComboBox()
        self.numberWidget.addItems(['1','2','3','4','5','6'])
        self.numberWidget.setCurrentIndex(0)
        self.numberWidget.currentIndexChanged.connect(self.changeNumberOfTypes)

        numlab = QtWidgets.QLabel("Number of types of oddballs")
        numlab.setAlignment(QtCore.Qt.AlignRight)
        numlab.setBuddy(self.numberWidget)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.chooseButton)
        hlayout.addStretch()

        hlayout.addWidget(numlab)
        hlayout.addWidget(self.numberWidget)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.tableWidget)

        self.setLayout(vlayout)

        self.chooseButton.clicked.connect(self.chooseOddballs)

        self.registerField("numberOfTypesIndex", self.numberWidget)

    def chooseOddballs(self):
        oddballFiles, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose oddball file(s)",
                                                              filter="Images (*.png *.jpg *.gif)")
        if oddballFiles is not None:
            self.tableWidget.clear()
            self.tableWidget.setRowCount(len(oddballFiles))
            for r, file1 in enumerate(oddballFiles):
                self.tableWidget.setItem(r, 0, QtWidgets.QTableWidgetItem(file1))
                self.tableWidget.setItem(r, 1, QtWidgets.QTableWidgetItem('1'))
            self.oddballFiles = oddballFiles

            self.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem("Filename"))
            self.tableWidget.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem("Type"))

    def changeNumberOfTypes(self, comboindex:int):
        if comboindex == 0:
            self.tableWidget.setColumnHidden(1, True)
            self.numtypes = 1
        else:
            self.tableWidget.setColumnHidden(1, False)
            self.numtypes = comboindex + 1

            self.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem("Filename"))
            self.tableWidget.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem("Type"))

    def getOddballTypes(self):
        if self.numtypes == 1:
            return [1] * len(self.oddballFiles)
        else:
            oddballTypes = []
            for row in range(self.tableWidget.rowCount()):
                twi = self.tableWidget.item(row, 1)
                try:
                    oddballTypes.append(int(twi.text()))
                except TypeError:
                    oddballTypes.append(1)

            return oddballTypes

    def validatePage(self):
        if self.numtypes > 1:
            oddballtypes = self.getOddballTypes()

            oddballtypes = set(oddballtypes)
            alltypes = set(range(1, self.numtypes+1))

            if len(alltypes - oddballtypes) > 0:
                ret = QtWidgets.QMessageBox.question(self, "Warning",
                                                      "You selected {} oddball types but only {} are in the table. "
                                                      "Is this what you meant?".format(self.numtypes, len(oddballtypes)),
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                return ret == QtWidgets.QMessageBox.Yes

        return True


class ChooseStandardPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super(ChooseStandardPage, self).__init__(parent)

        self.standardFiles = None

        self.setTitle('Choose standard images')
        self.setSubTitle('Choose the image or images that will be used for the common ("standard") stimulus')

        self.chooseButton = QtWidgets.QPushButton('Choose standard image(s)...', parent=self)
        self.listWidget = QtWidgets.QListWidget(parent=self)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.chooseButton)
        hlayout.addStretch()

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.listWidget)

        self.setLayout(vlayout)

        self.chooseButton.clicked.connect(self.chooseStandards)

    def chooseStandards(self):
        standardFiles, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose standard image(s)",
                                                              filter="Images (*.png *.jpg *.gif)")
        if standardFiles is not None:
            self.listWidget.clear()
            self.listWidget.addItems(standardFiles)
            self.standardFiles = standardFiles

class ParametersPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None, parameters=None):
        super(ParametersPage, self).__init__(parent)

        self.setTitle('Set parameters')
        self.setSubTitle('Set other parameters for the experiment')

        self.params = Parameter.create(name='Parameters', type='group',
                                       children = parameters)
        self.paramtree = ParameterTree()
        self.paramtree.setParameters(self.params, showTop=False)

        self.ntypes = 1

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.paramtree)

        self.params.child('Choose instructions image...').sigActivated.connect(self.chooseInstructionsImage)
        self.params.child('Choose cue image...').sigActivated.connect(self.chooseCueImage)
        self.params.child('Internals', 'Choose scratch directory...').sigActivated.connect(self.chooseScratchDirectory)

        self.setLayout(layout)

    def initializePage(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)
        settings.beginGroup("Parameters")

        self._readParameters(settings, self.params)

        settings.endGroup()

        logging.debug('more types! {}'.format(self.field("numberOfTypesIndex")))
        if self.field("numberOfTypesIndex")+1 != self.ntypes:
            oldntypes = self.ntypes
            self.ntypes = self.field("numberOfTypesIndex") + 1

            # if self.ntypes == 1:
            #     try:
            #         self.params.child('Number of each type of oddball').hide()
            #     except TypeError:
            #         # there's a bug in pyqtgraph so that this throws an exception, even though it actually works
            #         pass
            #     self.params.child('Number of each type of oddball').sigValueChanged.disconnect(self.updateTotalOddballs)
            #     self.params.child['Number of each type of oddball'] = 1
            #     try:
            #         self.params.child("Total number of oddballs").setWritable()
            #     except TypeError:
            #         # there's a bug in pyqtgraph so that this throws an exception, even though it actually works
            #         pass
            # else:
            #     try:
            #         self.params.child('Number of each type of oddball').show()
            #     except TypeError:
            #         # there's a bug in pyqtgraph so that this throws an exception, even though it actually works
            #         pass
            #     self.params.child('Number of each type of oddball').sigValueChanged.connect(self.updateTotalOddballs)
            #     try:
            #         self.params.child("Total number of oddballs").setReadonly()
            #     except TypeError:
            #         # there's a bug in pyqtgraph so that this throws an exception, even though it actually works
            #         pass

            if oldntypes > self.ntypes:
                for i in range(self.ntypes, oldntypes):
                    nm1 = "Oddball type {}".format(i + 1)
                    self.params.child(nm1).remove()

                if self.ntypes == 1:
                    self.params.child("Oddball type 1").setName("Oddball stimuli identifier")
            else:
                if oldntypes == 1:
                    self.params.child("Oddball stimuli identifier").remove()
                    startnew = 0
                else:
                    startnew = oldntypes

                next = self.params.child("Internals")
                for i in range(startnew, self.ntypes):
                    oddballbeep['name'] = "Oddball type {}".format(i + 1)
                    self.params.insertChild(next, oddballbeep)

    def chooseInstructionsImage(self):
        instrFile = self.params['Cue image']
        if not instrFile:
            instrFile = ""
        instrFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose output file", directory=instrFile,
                                                              filter="Images (*.png *.jpg *.gif)")
        if instrFile:
            self.params['Instructions image'] = instrFile

    def chooseCueImage(self):
        cueFile = self.params['Cue image']
        if not cueFile:
            cueFile = ""
        cueFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose output file", directory=cueFile,
                                                              filter="Images (*.png *.jpg *.gif)")
        if cueFile:
            self.params['Cue image'] = cueFile

    def chooseScratchDirectory(self):
        scratchDir = self.params['Internals', 'Scratch directory']
        if not scratchDir:
            scratchDir = ""
        scratchDir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose scratch directory",
                                                                   directory=scratchDir)
        if scratchDir:
            self.params['Internals', 'Scratch directory'] = scratchDir

    def updateTotalOddballs(self, val):
        self.params["Total number of oddballs per movie"] = self.ntypes * self.params["Number of each type of oddball"]

    def validatePage(self):
        if not os.path.exists(self.params['Cue image']):
            QtWidgets.QMessageBox.warning(self, "Error", "Cue image file does not exist. Please select it again")
            return False

        if not os.path.exists(self.params['Instructions image']):
            QtWidgets.QMessageBox.warning(self, "Error", "Instructions image file does not exist. Please select it again")
            return False

        if not os.path.exists(self.params['Instructions image']):
            QtWidgets.QMessageBox.warning(self, "Error", "Instructions image file does not exist. Please select it again")
            return False

        if not os.path.exists(self.params['Internals', 'Scratch directory']):
            QtWidgets.QMessageBox.warning(self, "Error", "Scratch directory does not exist. Please select it again")
            return False

        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)
        settings.beginGroup("Parameters")

        self._writeParameters(settings, self.params)

        settings.endGroup()

        return True

    def _writeParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                settings.setValue("Expanded", ch.opts['expanded'])
                self._writeParameters(settings, ch)
                settings.endGroup()
            elif ch.type() in ['float', 'int', 'list', 'str']:
                settings.setValue(ch.name(), ch.value())

    def _readParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                expanded = settings.value("Expanded", defaultValue=False)
                ch.setOpts(expanded=expanded)

                self._readParameters(settings, ch)
                settings.endGroup()
            else:
                if ch.type() == 'float':
                    if settings.contains(ch.name()):
                        v = settings.value(ch.name(), type=float)
                        ch.setValue(v)
                elif ch.type() == 'int':
                    if settings.contains(ch.name()):
                        v = settings.value(ch.name(), type=int)
                        ch.setValue(v)
                elif ch.type() in ['str', 'list']:
                    if settings.contains(ch.name()):
                        ch.setValue(settings.value(ch.name(), type=str))


def gcd(a, b, tol=1e-10):
    """ Greatest common divisor algorithm from Wikipedia.
    https://en.wikipedia.org/wiki/Euclidean_algorithm
    """
    while (b > tol) and (abs(b - a) > tol):
        t = b
        b = a % b
        a = t
    return a


def scale_image(qim, width, height):
    imout = QtGui.QImage(width, height, qim.format())
    imout.fill(QtGui.QColor(255, 255, 255))

    arin = qim.width() / qim.height()
    arout = width / height

    if arout > arin:
        scale = height / qim.height()
        scalewidth = qim.width() * scale
        scaleheight = height
    else:
        scale = width / qim.width()
        scaleheight = qim.height() * scale
        scalewidth = width

    qim = qim.scaled(scalewidth, scaleheight,
                     QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

    offsetx = (width - scalewidth)/2.0
    offsety = (height - scaleheight)/2.0

    painter = QtGui.QPainter()
    painter.begin(imout)
    painter.drawImage(offsetx, offsety, qim)
    painter.end()

    return imout

def main():
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    wizard = OddballWizard(parameters=parameters)
    wizard.exec_()

    _, framefmt = os.path.splitext(wizard.params['Internals', 'Frame name'])

    # convert image files to the right size and format
    instrcue = []
    for file in [wizard.params['Instructions image'], wizard.params['Cue image']]:
        if not os.path.exists(file):
            raise IOError('File {} not found'.format(file))

        qim = QtGui.QImage(file)

        logging.debug('qim.width = {}; qim.height = {}'.format(qim.width(), qim.height()))
        qim = scale_image(qim, wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'])

        _, fn = os.path.split(file)
        fn, _ = os.path.splitext(fn)
        fn1 = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
        qim.save(fn1)

        instrcue.append(fn1)

    instructionsfile, cuefile = instrcue

    oddfiles = []
    for file in wizard.oddballFiles:
        if not os.path.exists(file):
            raise IOError('File {} not found'.format(file))
        qim = QtGui.QImage(file)

        logging.debug('Qim.size before = {}'.format(qim.size()))
        qim = scale_image(qim, wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'])
        logging.debug('Qim.size = {}'.format(qim.size()))

        _, fn = os.path.split(file)
        fn, _ = os.path.splitext(fn)
        outfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
        qim.save(outfilename)

        oddfiles.append(outfilename)

    stdfiles = []
    for file in wizard.standardFiles:
        qim = QtGui.QImage(file)
        qim = scale_image(qim, wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'])

        _, fn = os.path.split(file)
        fn, _ = os.path.splitext(fn)
        outfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
        qim.save(outfilename)

        stdfiles.append(outfilename)

    instructionsdur = wizard.params['Instructions duration']

    cuedur = wizard.params['Duration of cue']
    stimdur = wizard.params['Duration of stimulus']
    mindur = gcd(cuedur, stimdur)

    instrframes = int(instructionsdur / mindur)
    cueframes = int(cuedur / mindur)
    stimframes = int(stimdur / mindur)

    basefps = 1.0 / mindur

    nodd = wizard.params['Total number of oddballs per movie']
    noddtypes = wizard.noddballTypes
    noddpertype = nodd / noddtypes

    ntotal = int(np.ceil(nodd / wizard.params['Oddball probability']))
    nstandard = ntotal - nodd

    totaldur = instructionsdur + ntotal * (cuedur + stimdur)

    oddfiles *= int(np.ceil(nodd / len(wizard.oddballFiles)))
    oddfiles = oddfiles[:nodd]

    oddtypes = wizard.oddballTypes
    oddtypes *= int(np.ceil(nodd / len(wizard.oddballFiles)))
    oddtypes = oddtypes[:nodd]

    stdfiles *= int(np.ceil(nstandard / len(wizard.standardFiles)))
    stdfiles = stdfiles[:nstandard]

    files0 = oddfiles + stdfiles
    files0 = np.array(files0)
    tstim = instructionsdur + cuedur + np.arange(0, ntotal)*(cuedur + stimdur)

    stimtype0 = np.concatenate((np.array(oddtypes, dtype=np.int),
                               np.zeros((nstandard,), dtype=np.int)))

    framename = os.path.join(wizard.params['Internals', 'Scratch directory'],
                              wizard.params['Internals', 'Frame name'])

    pn, _ = os.path.split(wizard.oddballFiles[0])
    outfilename, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Choose output file", directory=pn,
                                                           filter="Video (*.mp4)")
    if outfilename is None:
        return 0

    outfilebase, outfileext = os.path.splitext(outfilename)

    # shuffle the order of the files
    errors = []

    nreps = wizard.params['Number of movies to make']
    for rep in range(nreps):
        ord = np.random.permutation(ntotal)
        files = files0[ord]
        stimtype = stimtype0[ord]

        k = 1
        for fr in range(instrframes):
            fn = framename % k

            if os.path.exists(fn):
                os.unlink(fn)
            shutil.copyfile(instructionsfile, fn)

            k += 1

        for file in files:
            for f, nfr in zip([cuefile, file], [cueframes, stimframes]):
                for fr in range(nfr):
                    fn = framename % k

                    if os.path.exists(fn):
                        os.unlink(fn)
                    shutil.copyfile(f, fn)

                    k += 1

        # generate .wav file
        audiofreq = wizard.params['Internals', 'Audio rate']
        naudiosamp = int(np.ceil(totaldur * audiofreq))

        sounddata = np.zeros((naudiosamp,), dtype=np.int16)

        idstd = wizard.params.child('Standard stimuli identifier')

        if noddtypes == 1:
            idodd = [wizard.params.child('Oddball stimuli identifier'), ]
        else:
            idodd = [wizard.params.child("Oddball type {}".format(i + 1)) for i in range(noddtypes)]

        durs = [idstd['Duration']] + [x['Duration'] for x in idodd]
        notes = [idstd['Note']] + [x['Note'] for x in idodd]
        volumes = [idstd['Volume']] + [x['Volume'] for x in idodd]

        beep = []
        for dur, note, vol in zip(durs, notes, volumes):
            tbeep = np.arange(0, int(np.ceil(dur * audiofreq)), dtype=np.int16) / audiofreq
            try:
                beepfreq = NoteFrequencies[note]
            except KeyError:
                beepfreq = 440
            beep1 = 32767.0 * vol * np.sin(2*np.pi * beepfreq * tbeep)
            beep.append(beep1.astype(np.int16))

        for t1, stimtype1 in zip(tstim, stimtype):
            beep1 = beep[stimtype1]

            ind1 = int(np.round(t1 * audiofreq))
            k = ind1 + np.arange(len(beep1))

            try:
                sounddata[k] = beep1
            except IndexError:
                pass

        soundfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], 'id.wav')
        with wave.open(soundfilename, 'wb') as wav:
            wav.setframerate(audiofreq)
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.writeframes(sounddata)

        if nreps > 1:
            outfilename1 = '{}{:02d}{}'.format(outfilebase, rep+1, outfileext)
        else:
            outfilename1 = outfilename

        if os.path.exists(outfilename1):
            os.unlink(outfilename1)

        try:
            subprocess.run(['ffmpeg', '-r', str(basefps), '-i', framename,
                            '-i', soundfilename, '-crf', '23', '-r', '30',
                            '-c:a', 'aac', '-b:a', '192k', outfilename1],
                           check=True)
        except subprocess.CalledProcessError as err:
            logging.debug(err)
            errors.append(err)

    if len(errors) > 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    sys.exit(main())





import os, sys
import shutil
import logging

from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree

import numpy as np
import wave
import subprocess

SETTINGS_FILE = "oddball.ini"

# TODO: Add oddball classes

parameters = [
    {'name': 'Cue image', 'type': 'str', 'value': ''},
    {'name': 'Choose cue image...', 'type': 'action'},

    {'name': 'Oddball probability', 'type': 'float', 'value': 0.1,
     'limits': (0, 1)},
    {'name': 'Total number of oddballs', 'type': 'int', 'value': 10},
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

notes = {'A': 440, 'B': 494, 'C': 523, 'D': 587, 'E': 659, 'F': 698, 'G': 784}

class OddballWizard(QtWidgets.QWizard):
    def __init__(self, parent=None, parameters=None):
        super(OddballWizard, self).__init__(parent)

        self.oddballFiles = None
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
        self.listWidget = QtWidgets.QListWidget(parent=self)

        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(self.chooseButton)
        hlayout.addStretch()

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.listWidget)

        self.setLayout(vlayout)

        self.chooseButton.clicked.connect(self.chooseOddballs)

    def chooseOddballs(self):
        oddballFiles, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose oddball file(s)",
                                                              filter="Images (*.png *.jpg *.gif)")
        if oddballFiles is not None:
            self.listWidget.clear()
            self.listWidget.addItems(oddballFiles)
            self.oddballFiles = oddballFiles

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

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.paramtree)

        self.params.child('Choose cue image...').sigActivated.connect(self.chooseCueImage)
        self.params.child('Internals', 'Choose scratch directory...').sigActivated.connect(self.chooseScratchDirectory)

        self.setLayout(layout)

    def initializePage(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)
        settings.beginGroup("Parameters")

        self._readParameters(settings, self.params)

        settings.endGroup()

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

    def validatePage(self):
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
    ''' Greatest common divisor algorithm from Wikipedia.
    https://en.wikipedia.org/wiki/Euclidean_algorithm
    '''
    while (b > tol) and (abs(b - a) > tol):
        t = b
        b = a % b
        a = t
    return a

def main():
    logging.basicConfig(level=logging.DEBUG)

    app = QtWidgets.QApplication(sys.argv)
    wizard = OddballWizard(parameters=parameters)
    wizard.exec_()

    _, framefmt = os.path.splitext(wizard.params['Internals', 'Frame name'])

    # convert image files to the right size and format
    # TODO: Figure out how to scale images correctly without changing aspect ratio
    file = wizard.params['Cue image']
    qim = QtGui.QImage(file)

    if qim.width() != wizard.params['Internals', 'Image width'] or \
                    qim.height() != wizard.params['Internals', 'Image height']:
        qim = qim.scaled(wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'],
                         QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

    _, fn = os.path.split(file)
    fn, _ = os.path.splitext(fn)
    cuefile = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
    qim.save(cuefile)

    oddfiles = []
    for file in wizard.oddballFiles:
        qim = QtGui.QImage(file)

        if qim.width() != wizard.params['Internals', 'Image width'] or \
                        qim.height() != wizard.params['Internals', 'Image height']:
            logging.debug('Qim.size before = {}'.format(qim.size()))
            qim = qim.scaled(wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'],
                             QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            logging.debug('Qim.size = {}'.format(qim.size()))

        _, fn = os.path.split(file)
        fn, _ = os.path.splitext(fn)
        outfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
        qim.save(outfilename)

        oddfiles.append(outfilename)

    stdfiles = []
    for file in wizard.standardFiles:
        qim = QtGui.QImage(file)

        if qim.width() != wizard.params['Internals', 'Image width'] or \
                        qim.height() != wizard.params['Internals', 'Image height']:
            qim = qim.scaled(wizard.params['Internals', 'Image width'], wizard.params['Internals', 'Image height'],
                             QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

        _, fn = os.path.split(file)
        fn, _ = os.path.splitext(fn)
        outfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], fn + framefmt)
        qim.save(outfilename)

        stdfiles.append(outfilename)

    cuedur = wizard.params['Duration of cue']
    stimdur = wizard.params['Duration of stimulus']
    mindur = gcd(cuedur, stimdur)

    cueframes = int(cuedur / mindur)
    stimframes = int(stimdur / mindur)

    basefps = 1.0 / mindur

    nodd = wizard.params['Total number of oddballs']

    ntotal = int(np.ceil(nodd / wizard.params['Oddball probability']))
    nstandard = ntotal - nodd

    totaldur = ntotal * (cuedur + stimdur)

    oddfiles *= int(np.ceil(nodd / len(wizard.oddballFiles)))
    oddfiles = oddfiles[:nodd]

    stdfiles *= int(np.ceil(nstandard / len(wizard.standardFiles)))
    stdfiles = stdfiles[:nstandard]

    files = oddfiles + stdfiles
    files = np.array(files)
    tstim = cuedur + np.arange(0, ntotal)*(cuedur + stimdur)
    stimtype = np.concatenate((np.ones((nodd,), dtype=np.int), np.zeros((nstandard,), dtype=np.int)))

    framename = os.path.join(wizard.params['Internals', 'Scratch directory'],
                              wizard.params['Internals', 'Frame name'])

    # shuffle the order of the files
    ord = np.random.permutation(ntotal)
    files = files[ord]
    stimtype = stimtype[ord]

    k = 1
    for file in files:
        for f, nfr in zip([cuefile, file], [cueframes, stimframes]):
            for rep in range(nfr):
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
    idodd = wizard.params.child('Oddball stimuli identifier')

    beep = []
    for dur, note, vol in zip([idstd['Duration'], idodd['Duration']],
                              [idstd['Note'], idodd['Note']],
                              [idstd['Volume'], idodd['Volume']]):

        tbeep = np.arange(0, int(np.ceil(dur * audiofreq)), dtype=np.int16) / audiofreq
        try:
            beepfreq = notes[note]
        except KeyError:
            beepfreq = 440
        beep1 = 32767.0 * vol * np.sin(2*np.pi * beepfreq * tbeep)
        beep.append(beep1.astype(np.int16))

    for t1, stimtype1 in zip(tstim, stimtype):
        beep1 = beep[stimtype1]

        ind1 = int(np.round(t1 * audiofreq))
        k = ind1 + np.arange(len(beep1))

        sounddata[k] = beep1

    soundfilename = os.path.join(wizard.params['Internals', 'Scratch directory'], 'id.wav')
    with wave.open(soundfilename, 'wb') as wav:
        wav.setframerate(audiofreq)
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.writeframes(sounddata)

    pn, _ = os.path.split(wizard.oddballFiles[0])
    outfilename, _ = QtWidgets.QFileDialog.getSaveFileName(None, "Choose output file", directory=pn,
                                                          filter="Video (*.mp4)")
    if outfilename is None:
        return 0

    if os.path.exists(outfilename):
        os.unlink(outfilename)

    try:
        subprocess.run(['ffmpeg', '-r', str(basefps), '-i', framename,
                        '-i', soundfilename, '-c:v', 'libx264', '-crf', '23', '-r', '30',
                        '-c:a', 'aac', '-b:a', '192k', 'oddball.mp4'],
                       check=True)
    except subprocess.CalledProcessError as err:
        logging.debug(err)
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())





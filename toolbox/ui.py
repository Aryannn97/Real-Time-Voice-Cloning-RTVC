import sys
from pathlib import Path
from time import sleep
from typing import List, Set
from warnings import filterwarnings, warn

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import sounddevice as sd
import soundfile as sf
import umap
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from encoder.inference import plot_embedding_as_heatmap
from toolbox.utterance import Utterance

filterwarnings("ignore")

# Microphones whose names contain these words are virtual devices, which are not picked by default
virtual_microphone_words = ("ivcam", "steam streaming", "voicemod", "virtual", "stereo mix",
                            "sound mapper", "loopback", "obs", "primary sound")


# Look ---------------------------------------------------------------------------------------
# A recording-studio / spectrum-analyzer identity: warm paper, key-style buttons, an LCD readout
# for the log and a level-meter progress bar. The three model stages have their own colors.
BG = "#f1eee8"
PANEL = "#ffffff"
INK = "#1b1b1f"
MUTED = "#6f6a60"
LINE = "#cfc8ba"
KEY_EDGE = "#b3ab98"
SIGNAL = "#ff5a1f"
REC = "#d7263d"
LCD_BG = "#1d1d21"
LCD_TEXT = "#ffcf70"
STAGE_COLORS = {"encoder": "#0f8b8d", "synthesizer": "#b45309", "vocoder": "#9b2c5f"}

STYLESHEET = f"""
QDialog {{ background: {BG}; }}
QLabel {{ color: {INK}; font-size: 13px; }}
QLabel#heading {{ color: {INK}; font-family: Consolas, monospace; font-weight: 700;
    font-size: 12px; }}
QLabel#log {{ background: {LCD_BG}; border: 1px solid #000000; border-radius: 3px;
    padding: 8px 10px; color: {LCD_TEXT}; font-family: Consolas, monospace; font-size: 12px; }}
QPushButton {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-bottom: 3px solid {KEY_EDGE}; border-radius: 4px; padding: 5px 12px; font-size: 13px; }}
QPushButton:hover {{ border-color: {SIGNAL}; border-bottom-color: {SIGNAL}; }}
QPushButton:pressed {{ border-bottom-width: 1px; padding-top: 7px; background: #ebe7de; }}
QPushButton:disabled {{ color: #aaa497; background: #e9e5dc; border-color: #ddd7c9;
    border-bottom-color: #d3ccbc; }}
QPushButton#primary {{ background: {INK}; color: #ffffff; border: 1px solid {INK};
    border-bottom: 3px solid {SIGNAL}; font-weight: 700; font-size: 14px; padding: 8px 12px; }}
QPushButton#primary:hover {{ background: #2c2c33; }}
QPushButton#primary:pressed {{ border-bottom-width: 1px; padding-top: 10px; }}
QPushButton#primary:disabled {{ background: #8d8a84; color: #d8d4cb; border-color: #8d8a84;
    border-bottom-color: #b6b1a6; }}
QPushButton#record {{ background: {REC}; color: #ffffff; border: 1px solid #a51c2f;
    border-bottom: 3px solid #7c1424; font-weight: 700; }}
QPushButton#record:hover {{ background: #e63a50; }}
QPushButton#record:pressed {{ border-bottom-width: 1px; padding-top: 7px; }}
QPushButton#record:disabled {{ background: #e3a2ab; color: #ffffff; border-color: #d492a0;
    border-bottom-color: #c58795; }}
QComboBox, QLineEdit {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-radius: 3px; padding: 4px 8px; }}
QComboBox:hover, QLineEdit:focus {{ border-color: {SIGNAL}; }}
QComboBox:disabled, QLineEdit:disabled {{ color: #aaa497; background: #e9e5dc; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {INK};
    selection-background-color: {SIGNAL}; selection-color: white; }}
QPlainTextEdit {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-radius: 3px; padding: 8px; font-size: 15px; selection-background-color: {SIGNAL}; }}
QPlainTextEdit:focus {{ border-color: {SIGNAL}; }}
QCheckBox {{ color: {INK}; spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {KEY_EDGE}; border-radius: 2px;
    background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {SIGNAL}; border-color: {SIGNAL}; }}
QCheckBox:disabled {{ color: #aaa497; }}
QProgressBar {{ background: {LCD_BG}; border: 1px solid #000000; border-radius: 2px; height: 12px;
    text-align: center; color: #ffffff; }}
QProgressBar::chunk {{ border-radius: 1px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #3ddc84, stop:0.65 #ffd23f, stop:1 #ff5a1f); }}
QToolTip {{ background: {INK}; color: #ffffff; border: 1px solid {SIGNAL}; padding: 4px; }}
"""

# The plots follow the same colors; the warm "inferno" map suits the spectrum-analyzer look
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": LINE,
    "xtick.color": MUTED, "ytick.color": MUTED, "image.cmap": "inferno",
})

RECORD_LABEL = "●  Record 10 s"
WINDOW_TITLE = "Real-Time-Voice-Cloning"


def make_app_icon():
    """A small waveform badge, painted in code so no image file is needed."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(INK))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(2, 2, size - 4, size - 4, 12, 12)
    painter.setBrush(QColor(SIGNAL))
    for k, height in enumerate((14, 26, 40, 24, 34, 18, 10)):
        painter.drawRoundedRect(9 + k * 7, (size - height) // 2, 4, height, 2, 2)
    painter.end()
    return QIcon(pixmap)


colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float64) / 255

default_text = \
    "Hi, I'm testing the Real-Time-Voice-Cloning project. It listens to about ten seconds of " \
    "my speech, learns how I sound, and then it can say anything I type, in my own voice. " \
    "Can it really sound like me? Let's find out."


class UI(QDialog):
    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20

    def draw_utterance(self, utterance: Utterance, which):
        self.draw_spec(utterance.spec, which)
        self.draw_embed(utterance.embed, utterance.name, which)

    def draw_embed(self, embed, name, which):
        embed_ax, spec_ax = self.current_ax if which == "current" else self.gen_ax
        embed_ax.figure.suptitle("")
        if embed is not None and spec_ax.images:
            spec_ax.set_title("%s  ·  %s" % ("YOUR VOICE" if which == "current" else "CLONED VOICE",
                                             name), fontsize=10, color=MUTED)

        ## Embedding
        # Clear the plot
        if len(embed_ax.images) > 0:
            embed_ax.images[0].colorbar.remove()
        embed_ax.clear()

        # Draw the embed
        if embed is not None:
            plot_embedding_as_heatmap(embed, embed_ax)
            embed_ax.set_title("embedding")
        embed_ax.set_aspect("equal", "datalim")
        embed_ax.set_xticks([])
        embed_ax.set_yticks([])
        embed_ax.figure.canvas.draw()

    def draw_spec(self, spec, which):
        _, spec_ax = self.current_ax if which == "current" else self.gen_ax

        ## Spectrogram
        # Draw the spectrogram
        spec_ax.clear()
        if spec is not None:
            spec_ax.imshow(spec, aspect="auto", interpolation="none")
            spec_ax.set_title("%s  ·  mel spectrogram" %
                              ("YOUR VOICE" if which == "current" else "CLONED VOICE"),
                              fontsize=10, color=MUTED)

        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.figure.canvas.draw()
        if which != "current":
            self.vocode_button.setDisabled(spec is None)

    def draw_umap_projections(self, utterances: Set[Utterance]):
        self.umap_ax.clear()
        self.umap_ax.set_facecolor("#ffffff")

        speakers = np.unique([u.speaker_name for u in utterances])
        colors = {speaker_name: colormap[i] for i, speaker_name in enumerate(speakers)}
        embeds = [u.embed for u in utterances]

        # Display a message if there aren't enough points
        if len(utterances) < self.min_umap_points:
            self.draw_voice_map_placeholder(self.min_umap_points - len(utterances))
            self.umap_ax.set_title("")

        # Compute the projections
        else:
            if not self.umap_hot:
                self.log(
                    "Drawing UMAP projections for the first time, this will take a few seconds.")
                self.umap_hot = True

            reducer = umap.UMAP(int(np.ceil(np.sqrt(len(embeds)))), metric="cosine")
            projections = reducer.fit_transform(embeds)

            speakers_done = set()
            for projection, utterance in zip(projections, utterances):
                color = colors[utterance.speaker_name]
                mark = "x" if "_gen_" in utterance.name else "o"
                label = None if utterance.speaker_name in speakers_done else utterance.speaker_name
                speakers_done.add(utterance.speaker_name)
                self.umap_ax.scatter(projection[0], projection[1], c=[color], marker=mark,
                                     label=label)
            self.umap_ax.legend(prop={'size': 10})

        # Draw the plot. The placeholder artwork has fixed limits, so it keeps its shape by
        # resizing the axes box instead of stretching the data limits.
        self.umap_ax.set_aspect("equal", "box" if len(utterances) < self.min_umap_points else "datalim")
        self.umap_ax.set_xticks([])
        self.umap_ax.set_yticks([])
        self.umap_ax.figure.canvas.draw()

    def save_audio_file(self, wav, sample_rate):
        dialog = QFileDialog()
        dialog.setDefaultSuffix(".wav")
        fpath, _ = dialog.getSaveFileName(
            parent=self,
            caption="Select a path to save the audio file",
            filter="Audio Files (*.flac *.wav)"
        )
        if fpath:
            #Default format is wav
            if Path(fpath).suffix == "":
                fpath += ".wav"
            sf.write(fpath, wav, sample_rate)

    def setup_audio_devices(self, sample_rate):
        input_devices = []
        output_devices = []
        for device in sd.query_devices():
            # Check if valid input
            try:
                sd.check_input_settings(device=device["name"], samplerate=sample_rate)
                input_devices.append(device["name"])
            except:
                pass

            # Check if valid output
            try:
                sd.check_output_settings(device=device["name"], samplerate=sample_rate)
                output_devices.append(device["name"])
            except Exception as e:
                # Log a warning only if the device is not an input
                if not device["name"] in input_devices:
                    warn("Unsupported output device %s for the sample rate: %d \nError: %s" % (device["name"], sample_rate, str(e)))

        self.populate_microphones()

        if len(output_devices) == 0:
            self.log("No supported output audio devices were found! Audio output may not work.")
            self.audio_out_devices_cb.addItems(["None"])
            self.audio_out_devices_cb.setDisabled(True)
        else:
            self.audio_out_devices_cb.clear()
            self.audio_out_devices_cb.addItems(output_devices)
            self.audio_out_devices_cb.currentTextChanged.connect(self.set_audio_device)

        self.set_audio_device()

    def populate_microphones(self):
        """
        Lists the microphones and selects a real one. Virtual devices (phone-camera apps, voice
        changers, loopbacks) are skipped as the default because they often deliver silence.
        """
        self.audio_in_devices_cb.clear()
        try:
            host_api = sd.query_devices(kind="input")["hostapi"]
            microphones = [(device["name"], i) for i, device in enumerate(sd.query_devices())
                           if device["max_input_channels"] > 0 and device["hostapi"] == host_api]
        except Exception:
            microphones = []

        if len(microphones) == 0:
            self.log("No audio input device detected. Recording may not work.")
            self.audio_in_devices_cb.addItem("None", None)
            self.audio_in_devices_cb.setDisabled(True)
            return

        for name, index in microphones:
            self.audio_in_devices_cb.addItem(name, index)
        real = [i for i, (name, _) in enumerate(microphones)
                if not any(word in name.lower() for word in virtual_microphone_words)]
        self.audio_in_devices_cb.setCurrentIndex(real[0] if real else 0)
        self.audio_in_devices_cb.currentIndexChanged.connect(self.set_audio_device)

    def set_audio_device(self):

        output_device = self.audio_out_devices_cb.currentText()
        if output_device == "None":
            output_device = None
        input_device = self.audio_in_devices_cb.currentData()

        # If None, sounddevice queries portaudio
        sd.default.device = (input_device, output_device)

    def play(self, wav, sample_rate):
        try:
            sd.stop()
            sd.play(wav, sample_rate)
        except Exception as e:
            print(e)
            self.log("Error in audio playback. Try selecting a different audio output device.")
            self.log("Your device must be connected before you start the toolbox.")

    def stop(self):
        sd.stop()

    def record_one(self, sample_rate, duration):
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)

        self.log("Recording %d seconds from %s" % (duration, self.audio_in_devices_cb.currentText()))
        sd.stop()
        device = self.audio_in_devices_cb.currentData()
        try:
            # Record at the microphone's own rate, then resample for the models
            native_rate = int(sd.query_devices(device, "input")["default_samplerate"])
            wav = sd.rec(int(duration * native_rate), native_rate, 1, device=device)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            self.record_button.setText(RECORD_LABEL)
            self.record_button.setDisabled(False)
            return None

        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()

        self.log("Done recording.")
        self.record_button.setText(RECORD_LABEL)
        self.record_button.setDisabled(False)

        wav = wav.squeeze()
        if native_rate != sample_rate:
            wav = librosa.resample(wav, orig_sr=native_rate, target_sr=sample_rate)
        return wav

    @property
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()

    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()

    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        return Path(fpath[0]) if fpath[0] != "" else ""

    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)

    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Warning: you d" + ("id not pass a root directory for datasets as argument" \
                    if datasets_root is None else "o not have any of the recognized datasets" \
                                                  " in %s" % datasets_root)
                self.log(msg)
                msg += ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You " \
                       "can still use the toolbox by recording samples yourself." % \
                       ("\n\t".join(recognized_datasets))
                print(msg, file=sys.stderr)

                self.random_utterance_button.setDisabled(True)
                self.random_speaker_button.setDisabled(True)
                self.random_dataset_button.setDisabled(True)
                self.utterance_box.setDisabled(True)
                self.speaker_box.setDisabled(True)
                self.dataset_box.setDisabled(True)
                self.browser_load_button.setDisabled(True)
                self.auto_next_checkbox.setDisabled(True)
                return
            self.repopulate_box(self.dataset_box, datasets, random)

        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)

        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name,
                self.current_speaker_name
            )
            utterances = []
            for extension in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)

    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())

    @property
    def current_synthesizer_fpath(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())

    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())

    def populate_models(self, models_dir: Path):
        # Encoder
        encoder_fpaths = list(models_dir.glob("*/encoder.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % models_dir)
        self.repopulate_box(self.encoder_box, [(f.parent.name, f) for f in encoder_fpaths])

        # Synthesizer
        synthesizer_fpaths = list(models_dir.glob("*/synthesizer.pt"))
        if len(synthesizer_fpaths) == 0:
            raise Exception("No synthesizer models found in %s" % models_dir)
        self.repopulate_box(self.synthesizer_box, [(f.parent.name, f) for f in synthesizer_fpaths])

        # Vocoder
        vocoder_fpaths = list(models_dir.glob("*/vocoder.pt"))
        vocoder_items = [(f.parent.name, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.repopulate_box(self.vocoder_box, vocoder_items)

    @property
    def selected_utterance(self):
        return self.utterance_history.itemData(self.utterance_history.currentIndex())

    def register_utterance(self, utterance: Utterance):
        self.utterance_history.blockSignals(True)
        self.utterance_history.insertItem(0, utterance.name, utterance)
        self.utterance_history.setCurrentIndex(0)
        self.utterance_history.blockSignals(False)

        if len(self.utterance_history) > self.max_saved_utterances:
            self.utterance_history.removeItem(self.max_saved_utterances)

        self.play_button.setDisabled(False)
        self.generate_button.setDisabled(False)
        self.synthesize_button.setDisabled(False)

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)

        self.log_window.setText(log_text)
        self.app.processEvents()

    def set_loading(self, value, maximum=1):
        self.loading_bar.setValue(int(value * 100))
        self.loading_bar.setMaximum(int(maximum * 100))
        self.loading_bar.setTextVisible(value != 0)
        self.app.processEvents()

    def populate_gen_options(self, seed, trim_silences):
        if seed is not None:
            self.random_seed_checkbox.setChecked(True)
            self.seed_textbox.setText(str(seed))
            self.seed_textbox.setEnabled(True)
        else:
            self.random_seed_checkbox.setChecked(False)
            self.seed_textbox.setText(str(0))
            self.seed_textbox.setEnabled(False)

        if not trim_silences:
            self.trim_silences_checkbox.setChecked(False)
            self.trim_silences_checkbox.setDisabled(True)

    def update_seed_textbox(self):
        if self.random_seed_checkbox.isChecked():
            self.seed_textbox.setEnabled(True)
        else:
            self.seed_textbox.setEnabled(False)

    def reset_interface(self):
        self.draw_embed(None, None, "current")
        self.draw_embed(None, None, "generated")
        self.draw_spec(None, "current")
        self.draw_spec(None, "generated")
        self.draw_umap_projections(set())
        self.set_loading(0)
        self.play_button.setDisabled(True)
        self.generate_button.setDisabled(True)
        self.synthesize_button.setDisabled(True)
        self.vocode_button.setDisabled(True)
        self.replay_wav_button.setDisabled(True)
        self.export_wav_button.setDisabled(True)
        for line in ("Tip: press Record 10 s, then read the text on the right aloud or just talk about the project.",
                     "Use a laptop or wired microphone. Bluetooth headsets sound like a phone call.",
                     "Then press Synthesize and vocode to hear it in your voice."):
            self.log(line)

    @staticmethod
    def heading(text):
        label = QLabel(text.upper())
        label.setObjectName("heading")
        stage_color = STAGE_COLORS.get(text.strip(" :").lower())
        if stage_color:
            label.setStyleSheet("color: %s;" % stage_color)
        return label

    def draw_voice_map_placeholder(self, missing):
        """Artwork shown on the voice map until there are enough voices to project."""
        ax = self.umap_ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for k, radius in enumerate((0.075, 0.15, 0.225, 0.3)):
            ax.add_patch(Circle((0.5, 0.52), radius, fill=False, ec=SIGNAL, lw=2,
                                alpha=0.95 - 0.2 * k))
        ax.add_patch(Circle((0.5, 0.52), 0.035, color=SIGNAL))
        x = np.linspace(0.08, 0.92, 120)
        envelope = np.sin(np.linspace(0, np.pi, 120))
        ax.plot(x, 0.155 + 0.045 * envelope * np.sin(x * 90), color=INK, lw=1.2)
        ax.text(0.5, 0.955, "add %d more voice%s to see how\nclose they are" %
                (missing, "" if missing == 1 else "s"), ha="center", va="top", fontsize=10,
                color=MUTED)
        ax.text(0.5, 0.03, "VOICE MAP", ha="center", va="bottom", fontsize=9, color=INK,
                fontweight="bold", family="monospace")

    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(make_app_icon())
        self.app.setWindowIcon(make_app_icon())
        self.setStyleSheet(STYLESHEET)


        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)

        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 0, 1, 2)

        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2, 1, 2)

        # Projections
        self.projections_layout = QVBoxLayout()
        root_layout.addLayout(self.projections_layout, 1, 0, 1, 1)

        # Visualizations
        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 1, 3)


        ## Projections
        # UMap
        fig, self.umap_ax = plt.subplots(figsize=(3, 3), facecolor=BG)
        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.projections_layout.addWidget(FigureCanvas(fig))
        self.umap_hot = False
        self.clear_button = QPushButton("Clear")
        self.projections_layout.addWidget(self.clear_button)


        ## Browser
        # Dataset, speaker and utterance selection
        i = 0
        self.dataset_box = QComboBox()
        browser_layout.addWidget(self.heading("Dataset"), i, 0)
        browser_layout.addWidget(self.dataset_box, i + 1, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(self.heading("Speaker"), i, 1)
        browser_layout.addWidget(self.speaker_box, i + 1, 1)
        self.utterance_box = QComboBox()
        browser_layout.addWidget(self.heading("Utterance"), i, 2)
        browser_layout.addWidget(self.utterance_box, i + 1, 2)
        self.browser_load_button = QPushButton("Load")
        browser_layout.addWidget(self.browser_load_button, i + 1, 3)
        i += 2

        # Random buttons
        self.random_dataset_button = QPushButton("Random")
        browser_layout.addWidget(self.random_dataset_button, i, 0)
        self.random_speaker_button = QPushButton("Random")
        browser_layout.addWidget(self.random_speaker_button, i, 1)
        self.random_utterance_button = QPushButton("Random")
        browser_layout.addWidget(self.random_utterance_button, i, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        browser_layout.addWidget(self.auto_next_checkbox, i, 3)
        i += 1

        # Utterance box
        browser_layout.addWidget(self.heading("Use embedding from:"), i, 0)
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, i, 1, 1, 3)
        i += 1

        # Random & next utterance buttons
        self.browser_browse_button = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button, i, 0)
        self.record_button = QPushButton(RECORD_LABEL)
        self.record_button.setObjectName("record")
        self.record_button.setToolTip("Records 10 seconds from the microphone. Read the text on "
                                      "the right aloud, or just talk about the project.")
        browser_layout.addWidget(self.record_button, i, 1)
        self.play_button = QPushButton("▶  Play")
        browser_layout.addWidget(self.play_button, i, 2)
        self.stop_button = QPushButton("■  Stop")
        browser_layout.addWidget(self.stop_button, i, 3)
        i += 1


        # Model and audio output selection
        self.encoder_box = QComboBox()
        browser_layout.addWidget(self.heading("Encoder"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(self.heading("Synthesizer"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(self.heading("Vocoder"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)

        self.audio_out_devices_cb=QComboBox()
        browser_layout.addWidget(self.heading("Audio Output"), i, 3)
        browser_layout.addWidget(self.audio_out_devices_cb, i + 1, 3)
        i += 2

        # Microphone selection
        self.audio_in_devices_cb = QComboBox()
        browser_layout.addWidget(self.heading("Microphone:"), i, 0)
        browser_layout.addWidget(self.audio_in_devices_cb, i, 1, 1, 3)
        i += 1

        #Replay & Save Audio
        browser_layout.addWidget(self.heading("Toolbox Output:"), i, 0)
        self.waves_cb = QComboBox()
        self.waves_cb_model = QStringListModel()
        self.waves_cb.setModel(self.waves_cb_model)
        self.waves_cb.setToolTip("Select one of the last generated waves in this section for replaying or exporting")
        browser_layout.addWidget(self.waves_cb, i, 1)
        self.replay_wav_button = QPushButton("▶  Replay")
        self.replay_wav_button.setToolTip("Replay last generated vocoder")
        browser_layout.addWidget(self.replay_wav_button, i, 2)
        self.export_wav_button = QPushButton("⭳  Export")
        self.export_wav_button.setToolTip("Save last generated vocoder audio in filesystem as a wav file")
        browser_layout.addWidget(self.export_wav_button, i, 3)
        i += 1


        ## Embed & spectrograms
        vis_layout.addStretch()

        gridspec_kw = {"width_ratios": [1, 4]}
        fig, self.current_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor=BG,
                                            gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        fig, self.gen_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor=BG,
                                        gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        for ax in self.current_ax.tolist() + self.gen_ax.tolist():
            ax.set_facecolor(BG)
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)


        ## Generation
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)

        self.generate_button = QPushButton("Synthesize and vocode")
        self.generate_button.setObjectName("primary")
        gen_layout.addWidget(self.generate_button)

        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only")
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)

        layout_seed = QGridLayout()
        self.random_seed_checkbox = QCheckBox("Random seed:")
        self.random_seed_checkbox.setToolTip("When checked, makes the synthesizer and vocoder deterministic.")
        layout_seed.addWidget(self.random_seed_checkbox, 0, 0)
        self.seed_textbox = QLineEdit()
        self.seed_textbox.setMaximumWidth(80)
        layout_seed.addWidget(self.seed_textbox, 0, 1)
        self.trim_silences_checkbox = QCheckBox("Enhance vocoder output")
        self.trim_silences_checkbox.setToolTip("When checked, trims excess silence in vocoder output."
            " This feature requires `webrtcvad` to be installed.")
        layout_seed.addWidget(self.trim_silences_checkbox, 0, 2, 1, 2)
        self.clean_audio_checkbox = QCheckBox("Clean up audio")
        self.clean_audio_checkbox.setChecked(True)
        self.clean_audio_checkbox.setToolTip("When checked, removes hiss and rumble from the "
            "generated audio and evens out its loudness.")
        layout_seed.addWidget(self.clean_audio_checkbox, 1, 0, 1, 2)
        gen_layout.addLayout(layout_seed)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)

        self.log_window = QLabel()
        self.log_window.setObjectName("log")
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()


        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.8
        self.resize(max_size)

        ## Finalize the display
        self.reset_interface()
        self.show()

    def start(self):
        self.app.exec_()

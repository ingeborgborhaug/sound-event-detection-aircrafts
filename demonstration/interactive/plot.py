import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import tensorflow as tf

import sounddevice as sd
import soundfile as sf
from keras_yamnet import params
import time



class Plotter():
    def __init__(self, n_classes, win_size, n_wins, spec, pred, waveform, sr, start, end, n_bands=64, msd_labels=None, FIG_SIZE=(8,8),blit=True):
        # initialize plots

        self.wav_data , self.sr = waveform, sr
        duration = (n_wins * params.PATCH_HOP_SECONDS) 
        print(f'Duration of audio level right before plot: {duration} seconds')


        self.fps = 10 # How often moving line is updated
        self.duration = duration

        self.spec = spec # np.zeros((n_bands, win_size*n_wins))
        self.act = pred # np.zeros((n_classes, n_wins))
        #self.ref = 

        self.blit=blit
        self.win_size = win_size
        self.n_wins = n_wins # Eqal to number of predictions in plot
        self.n_bands = n_bands 
        self.n_classes = n_classes
        self.msd_labels = msd_labels

        # --- Set up the figure and axes ---
        # Make the figure a bit wider to ensure space for the colorbar
        self.fig, self.axs = plt.subplots(3, 1, sharex=True, figsize=(13, 8), height_ratios=[1, 1, 0.2])

        # Plot spectrogram
        img1 = self.axs[0].imshow(self.spec, aspect='auto', origin='lower',
                            extent=[0, duration, 0, n_bands], cmap='magma')
        self.axs[0].set_ylabel('Mel Bands')


        img2 = self.axs[1].imshow(tf.transpose(self.act), aspect='auto', origin='lower',
                            extent=[0, duration, -0.5, n_classes-0.5], cmap='viridis')
        self.axs[1].set_ylabel('Prediction')

        """ img3 = self.axs[2].imshow(tf.transpose(self.act), aspect='auto', origin='lower',
                          extent=[0, duration, -0.5, n_classes - 0.5], cmap='viridis')
        self.axs[2].set_ylabel('Reference') """

        # Add a small colorbar for class prediction values in the top left white space
        # [left, bottom, width, height] in figure coordinates (0,0 is bottom left)
        cbar_width = 0.018
        cbar_height = 0.25
        cbar_left = 0.08
        cbar_bottom = 0.67
        cax = self.fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cb = self.fig.colorbar(img2, cax=cax, orientation='vertical', label='Activation')
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')

        if msd_labels is not None:
            self.axs[1].set_yticks(np.arange(len(msd_labels)))
            self.axs[1].set_yticklabels(msd_labels)
            self.axs[1].set_ylim(-0.5, len(msd_labels)-0.5)

        # Playback bar with time labels
        self.axs[2].barh(0.5, duration, height=1, color='lightgray')
        self.axs[2].set_xlim(0, duration)
        self.axs[2].set_ylim(0, 1)
        self.axs[2].set_yticks([])
        ticks = np.linspace(0, duration, int(duration) + 1)
        self.axs[2].set_xticks(ticks)
        self.axs[2].set_xticklabels([f'{t:.1f}s' for t in ticks])
        self.axs[2].axis('on')

        # Shared playback vertical line
        self.playback_line1 = self.axs[0].axvline(0, color='red')
        self.playback_line2 = self.axs[1].axvline(0, color='red')
        self.playback_line3 = self.axs[2].axvline(0, color='red')

        # Interactive controls
        self.paused = False
        self.current_time = 0.0  # seconds
        self.playback_start_walltime = None  # wall-clock time when playback started
        self.ani = FuncAnimation(self.fig, self.update, interval=1000/self.fps, blit=True)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)


        # --- Animate ---

        plt.tight_layout()
        if not self.paused:
            self.start_playback(0.0)
        plt.show()


    def update(self, frame):
        if not self.paused and self.playback_start_walltime is not None:
            # Calculate elapsed time since playback started
            elapsed = time.time() - self.playback_start_walltime
            self.current_time = min(elapsed, self.duration)
            if self.current_time >= self.duration:
                self.paused = True
                sd.stop()
        t = self.current_time
        for line in (self.playback_line1, self.playback_line2, self.playback_line3):
            line.set_xdata([t])
        return self.playback_line1, self.playback_line2, self.playback_line3

    def onclick(self, event):
        # Skip to clicked time on playback bar
        if event.inaxes == self.axs[2]:
            seek_time = max(0.0, min(event.xdata, self.duration))
            self.current_time = seek_time
            if not self.paused:
                self.start_playback(seek_time)
            else:
                self.play_audio_from_time(seek_time, stop=True)

    def onkey(self, event):
        if event.key == ' ':
            self.paused = not self.paused  # Space to pause/resume
            if not self.paused:
                self.start_playback(self.current_time)
            else:
                sd.stop()
        elif event.key == 'right':
            seek_time = min(self.duration, self.current_time + 1.0)
            self.current_time = seek_time
            if not self.paused:
                self.start_playback(seek_time)
            else:
                self.play_audio_from_time(seek_time, stop=True)
        elif event.key == 'left':
            seek_time = max(0.0, self.current_time - 1.0)
            self.current_time = seek_time
            if not self.paused:
                self.start_playback(seek_time)
            else:
                self.play_audio_from_time(seek_time, stop=True)


    def start_playback(self, start_time):
        # Start playback from a given time and sync wall-clock
        self.playback_start_walltime = time.time() - start_time
        self.play_audio_from_time(start_time)

    def play_audio_from_time(self, start_time, stop=False):
        start_sample = int(start_time * self.sr)
        if stop:
            sd.stop()
        else:
            sd.stop()
            sd.play(self.wav_data[start_sample:], self.sr)

    def play_audio(self):
        sd.play(self.wav_data, self.sr)


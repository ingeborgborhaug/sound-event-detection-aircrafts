import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import tensorflow as tf

import sounddevice as sd
from keras_yamnet import params
import time
from matplotlib import gridspec
from matplotlib.transforms import Bbox
from matplotlib.colors import ListedColormap, BoundaryNorm
import soundfile as sf
from keras_yamnet.preprocessing import preprocess_input
from dataset import gt_conversion_functions as cf

import os
import pickle


class Plotter():
    def __init__(self, n_classes, wavfile, starttime, model, endtime, gt=None, n_bands=64, msd_labels=None, FIG_SIZE=(8,8),blit=True, save_pdf_path=None):

        duration = ((endtime-starttime) // params.PATCH_HOP_SECONDS) * params.PATCH_HOP_SECONDS
        print(f'Duration of audio level right before plot: {duration} seconds')
        self.duration = duration

        # Load waveform from start to end time
        info = sf.info(wavfile)
        sr = info.samplerate
        self.wav_data, self.sr = sf.read(wavfile, start=starttime*sr, stop=endtime*sr, dtype='int16')

        # Get prediction and spectrogram for waveform
        variables = self.process_and_cache(wavfile, self.wav_data, self.sr, model, force=True)
        
        self.pred = variables['prediction'] #[cf.sec_to_start_index(starttime):cf.sec_to_start_index(endtime + duration), :]

        #frame_start = int(round(starttime / params.STFT_HOP_SECONDS))
        #frame_end = int(round((starttime + duration) / params.STFT_HOP_SECONDS))
        self.spec = variables['spectrogram'] #[frame_start:frame_end]

        print(f'Duration of spectrogram level right before plot: {self.spec.shape[0] * params.STFT_HOP_SECONDS} seconds')

        self.starttime = float(starttime)
        self.endtime = self.starttime + self.duration

        self.fps = 10 # How often moving line is updated

        self.act = self.pred # np.zeros((n_classes, n_wins))

        if gt is None:
            self.gt = np.zeros((n_wins,), dtype=float)
            print("No ground truth provided, using zeros.")
        else:
            self.gt = gt # np.zeros((n_wins,), dtype=float)

        self.blit=blit
        self.n_bands = n_bands 
        self.n_classes = n_classes
        self.msd_labels = msd_labels

        # --- Set up the figure and axes ---
        # Make the figure a bit wider to ensure space for the colorbar
        self.fig = plt.figure(figsize=(13, 8))
        prediction_height = 0.14 if n_classes == 1 else 0.3
        gs = gridspec.GridSpec(4, 2, width_ratios=[50, 1], height_ratios=[1, prediction_height, 0.14, 0.2], wspace=0.05)

        self.axs = [self.fig.add_subplot(gs[i, 0]) for i in range(4)]

        # Plot spectrogram as raster image (kept non-vector by request)
        img1 = self.axs[0].imshow(
            self.spec,
            aspect='auto',
            origin='lower',
            extent=[self.starttime, self.endtime, 0, n_bands],
            cmap='magma',
            rasterized=True,
        )
        self.axs[0].set_ylabel('Mel Bands')
        self.axs[0].set_xlim(self.starttime, self.endtime)
        self.axs[0].set_ylim(0, n_bands)


        if n_classes == 1:
            prediction_plot = np.asarray(self.act).reshape(-1)
            pred_x = np.linspace(self.starttime, self.endtime, prediction_plot.shape[0] + 1)
            pred_y = np.array([-0.5, 0.5])
            img2 = self.axs[1].pcolormesh(
                pred_x,
                pred_y,
                prediction_plot[np.newaxis, :],
                cmap='gray',
                vmin=0,
                vmax=1,
                shading='flat',
            )
            self.axs[1].set_yticks([])
            self.axs[1].set_xlim(self.starttime, self.endtime)
            self.axs[1].set_ylim(-0.5, 0.5)
        else:
            pred_values = np.asarray(tf.transpose(self.act))
            pred_x = np.linspace(self.starttime, self.endtime, pred_values.shape[1] + 1)
            pred_y = np.arange(-0.5, n_classes + 0.5, 1.0)
            img2 = self.axs[1].pcolormesh(
                pred_x,
                pred_y,
                pred_values,
                cmap='gray',
                vmin=0,
                vmax=1,
                shading='flat',
            )
            self.axs[1].set_xlim(self.starttime, self.endtime)
            self.axs[1].set_ylim(-0.5, n_classes - 0.5)
        self.axs[1].set_ylabel('Prediction', rotation=0, ha='right', va='center', labelpad=12)

        gt_cmap = ListedColormap(['black', 'white'])
        gt_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=gt_cmap.N)

        gt_plot = np.asarray(self.gt).reshape(-1)
        gt_x = np.linspace(self.starttime, self.endtime, gt_plot.shape[0] + 1)
        gt_y = np.array([-0.5, 0.5])
        img3 = self.axs[2].pcolormesh(
            gt_x,
            gt_y,
            gt_plot[np.newaxis, :],
            cmap=gt_cmap,
            norm=gt_norm,
            shading='flat',
        )
        self.axs[2].set_ylabel('Ground truth', rotation=0, ha='right', va='center', labelpad=12)
        self.axs[2].set_yticks([])
        self.axs[2].set_xlim(self.starttime, self.endtime)
        self.axs[2].set_ylim(-0.5, 0.5)

        # Add a small colorbar for class prediction values in the top left white space
        # [left, bottom, width, height] in figure coordinates (0,0 is bottom left)
        self.cax = self.fig.add_subplot(gs[1, 1])  # spans top two rows, right side
        cb = self.fig.colorbar(img2, cax=self.cax, orientation='vertical')

        self.cax.yaxis.set_ticks_position('right')
        self.cax.yaxis.set_label_position('right')

        self.gt_cax = self.fig.add_subplot(gs[2, 1])
        gt_cb = self.fig.colorbar(img3, cax=self.gt_cax, orientation='vertical')
        gt_cb.set_ticks([0, 1])
        gt_cb.set_ticklabels(['0', '1'])
        self.gt_cax.yaxis.set_ticks_position('right')
        self.gt_cax.yaxis.set_label_position('right')

        if msd_labels is not None:
            self.axs[1].set_yticks(np.arange(len(msd_labels)))
            self.axs[1].set_yticklabels(msd_labels)
            self.axs[1].set_ylim(-0.5, len(msd_labels)-0.5)

        # Playback bar with time labels
        self.axs[3].barh(0.5, self.duration, left=self.starttime, height=1, color='lightgray')
        self.axs[3].set_xlim(self.starttime, self.endtime)
        self.axs[3].set_ylim(0, 1)
        self.axs[3].set_yticks([])
        ticks = np.arange(self.starttime, self.endtime + 1, 5)
        self.axs[0].set_xticks(ticks)
        self.axs[0].set_xticklabels([f'{t:.1f}s' for t in ticks])
        self.axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        self.axs[1].set_xticks(ticks)
        self.axs[1].set_xticklabels([f'{t:.1f}s' for t in ticks])
        self.axs[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        self.axs[2].set_xticks(ticks)
        self.axs[2].set_xticklabels([f'{t:.1f}s' for t in ticks])
        self.axs[2].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        self.axs[2].set_xlabel('Time')
        self.axs[3].set_xticks([])
        self.axs[3].set_xticklabels([])
        self.axs[3].axis('on')

        # Shared playback vertical line
        self.playback_line1 = self.axs[0].axvline(self.starttime, color='red')
        self.playback_line2 = self.axs[1].axvline(self.starttime, color='red')
        self.playback_line3 = self.axs[2].axvline(self.starttime, color='red')
        self.playback_line4 = self.axs[3].axvline(self.starttime, color='red')

        # Interactive controls
        self.paused = False
        self.current_time = 0.0  # seconds
        self.playback_start_walltime = None  # wall-clock time when playback started
        self.ani = FuncAnimation(self.fig, self.update, interval=1000/self.fps, blit=True)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)


        # --- Animate ---

        plt.tight_layout()
        if save_pdf_path is not None:
            self.save_pdf_without_playback(save_pdf_path)
        if not self.paused:
            self.start_playback(0.0)
        plt.show()

    def process_and_cache(self, audio_path, audio_wave, sample_rate, model, force=False):

        cache_file = os.path.splitext(audio_path)[0] + '_demo' + '.pkl'

        if os.path.exists(cache_file) and not force:
            print(f"Loading cached result for {audio_path}")
            with open(cache_file, 'rb') as f:
                variables = pickle.load(f)
        else:
            print(f'Processing and caching: {audio_path}')

            data_patches, spectrogram = preprocess_input(audio_wave, sample_rate)

            plt.figure()
            plt.imshow(spectrogram, aspect='auto', origin='lower')
            plt.title('Spectrogram for debugging')

            prediction = model.predict(data_patches)
            #prediction = prediction.detach().cpu().numpy()  # <-- Add this line


            variables = {
                "prediction": prediction,
                "spectrogram": spectrogram
            }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(variables, f)
        
        print(f"Cached result saved to {cache_file}")
    
        return variables

    def save_pdf_without_playback(self, output_path):
        line_visibility = [
            self.playback_line1.get_visible(),
            self.playback_line2.get_visible(),
            self.playback_line3.get_visible(),
            self.playback_line4.get_visible(),
        ]
        playback_bar_visible = self.axs[3].get_visible()

        try:
            self.playback_line1.set_visible(False)
            self.playback_line2.set_visible(False)
            self.playback_line3.set_visible(False)
            self.playback_line4.set_visible(False)
            self.axs[3].set_visible(False)

            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            bboxes = [
                self.axs[0].get_tightbbox(renderer),
                self.axs[1].get_tightbbox(renderer),
                self.axs[2].get_tightbbox(renderer),
                self.cax.get_tightbbox(renderer),
                self.gt_cax.get_tightbbox(renderer),
            ]
            bboxes = [bbox for bbox in bboxes if bbox is not None]
            export_bbox = Bbox.union(bboxes).transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(output_path, format='pdf', bbox_inches=export_bbox)
            print(f"Saved figure to {output_path}")
        finally:
            self.playback_line1.set_visible(line_visibility[0])
            self.playback_line2.set_visible(line_visibility[1])
            self.playback_line3.set_visible(line_visibility[2])
            self.playback_line4.set_visible(line_visibility[3])
            self.axs[3].set_visible(playback_bar_visible)
            self.fig.canvas.draw_idle()


    def update(self, frame):
        if not self.paused and self.playback_start_walltime is not None:
            # Calculate elapsed time since playback started
            elapsed = time.time() - self.playback_start_walltime
            self.current_time = min(elapsed, self.duration)
            if self.current_time >= self.duration:
                self.paused = True
                sd.stop()
        t = self.starttime + self.current_time
        for line in (self.playback_line1, self.playback_line2, self.playback_line3, self.playback_line4):
            line.set_xdata([t])
        return self.playback_line1, self.playback_line2, self.playback_line3, self.playback_line4

    def onclick(self, event):
        # Skip to clicked time on playback bar
        if event.inaxes == self.axs[3]:
            seek_time_abs = max(self.starttime, min(event.xdata, self.endtime))
            seek_time = seek_time_abs - self.starttime
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

